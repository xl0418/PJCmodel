#include <iostream>
#include <fstream>
#include <stdexcept>
#include <mutex>
#include <set>
#include <vector>
#include <chrono>
#include <tbb/tbb.h>
#include "jc.h"
#include "jc_simd.h"
#include "observer.hpp"
#include "cmd_line.h"



const char* JCHelp = R"(Usage: jc [OPTIONS]... PARAMETER...
Options:
  --help           prints this text and exits
  --version        prints version information and exits
  -v, --verbose    verbose output to console
  --check          check parameter set, no execution
  --torus          use periodic boundary
  --profile        create profile file(s)
  threads          number of threads

Model parameter:
  L                grid size
  L0               optional initial area size, clamped to L
  Linc             optional L increment, defaults to 1
  Lg               area increment L <- L + Linc every Lg years
  ticks            time steps [y]
  mu               1/average life time [1/y], G = mu * L * L
  nu               speciation rate, Nu = nu * G
  s_nu             optional sigma speciation radius, defaults to L/4
  Psi              strength of PJC effect
  Phi	             size of phylogenetic effect
  s_jc             width of PJC effect wrt spatial distance, use '-1' for 'infinite'
  jc_cutoff        max. distance spatial JC kernel.
  s_disp           width of dispersal kernel, use '-1' for 'infinite'
  disp_cutoff      max. distance dispersal kernel.

Output Parameter:
  log              optional log interval [events] for D and R, defaults to last state
  continue         result file of former simulation to be continued
  file             data file path

Alternative:
  batch            parameter file path, a file that contains multiple parameter sets
)";


// parse model parameter
auto ParseParameter(cmd::cmd_line_parser&& cmd, jc::Parameter paramIn)
{
  auto param = paramIn;
  param.L = cmd.required<int>("L");
  param.L0 = param.L;
  if (cmd.optional("L0", param.L0)) {
    param.L0 = std::min(param.L0, param.L);
    cmd.optional("Linc", param.Linc);
    param.Lg = cmd.required<int>("Lg");
    if (param.Lg <= 0) throw cmd::parse_error("invalid Lg");
  }
  param.L0 = std::min(param.L0, param.L);
  param.mu = cmd.required<double>("mu");
  param.nu = cmd.required<double>("nu");
  param.s_nu = param.L / 4;
  cmd.optional("s_nu", param.s_nu);
  param.Psi = cmd.required<double>("Psi");
  if (param.Psi <= 0.0)
  { // neutral model
    param.Phi = -1;
    param.s_jc = -1;
    param.s_disp = -1;
    cmd.recognize("Phi");
    cmd.recognize("s_jc");
    cmd.recognize("jc_cutoff");
    if (cmd.optional("s_disp", param.s_disp)) {
      param.disp_cutoff = cmd.required<int>("disp_cutoff");
    }
  }
  else {
    param.Phi = cmd.required<double>("Phi");
    if (param.Phi < 0.0) throw cmd::parse_error("invalid Phi");
    if (cmd.optional("s_jc", param.s_jc)) {
      if (param.s_jc <= 0.0) throw cmd::parse_error("invalid s_jc");
      param.jc_cutoff = cmd.required<int>("jc_cutoff");
      if ((param.jc_cutoff <= 0) || (param.jc_cutoff > param.L / 2)) throw cmd::parse_error("invalid jc_cutoff");
    }
    if (cmd.optional("s_disp", param.s_disp)) {
      if (param.s_disp <= 0.0) throw cmd::parse_error("invalid s_disp");
      param.disp_cutoff = cmd.required<int>("disp_cutoff");
      if ((param.disp_cutoff <= 0) || (param.disp_cutoff > param.L / 2)) throw cmd::parse_error("invalid disp_cutoff");
    }
  }
  param.ticks = static_cast<int64_t>(cmd.required<double>("ticks"));
  cmd.optional("log", param.log);
  if (!cmd.optional("file", param.filename)) {
    std::cout << "'file' argument not given. No output will be produced\n";
  }
  if (cmd.optional("continue", param.cont)) {
    auto is = std::ifstream(param.cont);
    if (!is) throw cmd::parse_error("continuation file doesn't exists or is corrupted");
  }
  param.torus = cmd.flag("--torus");
  if (!param.torus) {
    throw cmd::parse_error("grid is deprecated, please use --torus");
  }
  if (auto ur = cmd.unrecognized(); !ur.empty()) throw cmd::parse_error((std::string("invalid argument \'") + ur[0] + '\'').c_str());
  if (param.L0 != param.L) {
    // check burnin settings
    if ((param.L0 < param.jc_cutoff) || (param.L0 < param.disp_cutoff)) throw cmd::parse_error("Invalid L0");
  }
  return param;
}


std::vector<jc::Parameter> queue;
size_t remaining_runs = 0;


void RunModel(const jc::Parameter& CliParam, const jc::Parameter& param)
{
  auto obs = std::make_unique<jc::Observer>();
  if (CliParam.verbose) {
    obs = std::move(jc::create_cmd_line_obs(param, std::move(obs)));
  }
  if (CliParam.summary) {
    obs = std::move(jc::create_summary_obs(param, std::move(obs)));
  }
  if (!param.filename.empty()) {
    obs = std::move(jc::create_log_obs(param, std::move(obs)));
  }
  if (param.profile) {
    obs = std::move(jc::create_profile_obs(param, std::move(obs)));
  }
  if (param.checkInterval) {
    obs = std::move(jc::create_check_obs(param, std::move(obs)));
  }
  if (!param.cont.empty()) {
    obs->notify(jc::ObserverMsg::ParsingContinuation, 0, nullptr, nullptr);
  }
  auto model = std::make_unique<jc::Model>(param);
  model->run(obs.get());
}


void RunBatch(const jc::Parameter& CliParam)
{
  remaining_runs = queue.size();
  for (size_t i = 0; i < queue.size(); ++i) {
    queue[i].simId = static_cast<int>(i);
    RunModel(CliParam, queue[i]);
  }
}


int main(int argc, const char** argv)
{
  try {
    cmd::cmd_line_parser cmd(argc, argv);
    if (cmd.flag("--help")) {
      std::cout << JCHelp << std::endl;
      return 0;
    }
    if (cmd.flag("--version")) {
      std::cout << jc::Version << '\n';
      std::cout << jc::ReportBugs << std::endl;
      return 0;
    }
    jc::Parameter param;
#ifdef JC_DEBUG
    cmd.recognize("threads");
    param.nthreads = 1;
#else
    if (!cmd.optional("threads", param.nthreads)) {
      param.nthreads = static_cast<int>(std::thread::hardware_concurrency());
    }
#endif
    tbb::global_control c(tbb::global_control::max_allowed_parallelism, param.nthreads);
    bool check = cmd.flag("--check");
    param.verbose = cmd.flag("-v") || cmd.flag("--verbose");
    param.profile = cmd.flag("--profile");
    param.summary = cmd.flag("--summary");
    cmd.optional("batch", param.batch);
    cmd.optional("check", param.checkInterval);
    if (param.batch.empty()) { 
      // parameter given in command line
      queue.emplace_back(ParseParameter(std::move(cmd), param));
    }
    else { 
      // parameter given in text file
      std::ifstream is(param.batch.c_str());
      if (!is) throw std::runtime_error(param.batch + " doesn't exists");
      std::set<std::string> files;  // check unique file names
      while (is) {
        std::string cl;
        std::getline(is, cl);
        if (!cl.empty()) {
          auto mp = ParseParameter(cmd::merge(cmd, cmd::cmd_line_parser(cl)), param);
          mp.verbose = mp.profile = false;  // no messages from enqueued tasks, please
          if (!files.insert(mp.filename).second) { 
            // sanity check
            throw cmd::parse_error((param.batch + ": duplicated file name \'" + mp.filename + '\'').c_str());
          }
          queue.emplace_back(mp);
        }
      }
    }
    if (check) {
      std::cout << "check passed.\n";
      return 0;
    }
    // run everything in queue
    RunBatch(param);
    return 0;
  }
  catch (cmd::parse_error& e) {
    std::cerr << "jc: fatal error: " << e.what() << std::endl;
    std::cout << "try jc --help\n";
  }
  catch (std::exception& e) {
    std::cerr << "jc: fatal error: " << e.what() << std::endl;
  }
  catch (...) {
    std::cerr << "jc: fatal error: unknown exception" << std::endl;
  }
  return -1;
}
