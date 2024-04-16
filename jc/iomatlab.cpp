#include <stdexcept>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <string>
#include "iomatlab.h"
#include "jc.h"
#include <iostream>


namespace jc {

  const char EventHeader[] = R"m(
%
% Speciation event log
%
% events(:,1)    The time the event occurred.
% events(:,2)    Number of species after the event.
% events(:,3:4)  Position on the grid.
% events(:,5)    New or extinct species.
% events(:,6)    Ancestor in case of speciation event, -1 otherwise.
%
% extinctions    Extinction events from events
% speciations    Speciation events from events
%
)m";


  std::ostream& operator<<(std::ostream& os, const event& e)
  {
    os << "  " << e.T << ' ' << e.NS << ' ' << e.pos.x << ' ' << e.pos.y << ' ' << e.sp << ' ' << e.ancestor << ';';
    return os;
  }


  std::istream& operator>>(std::istream& is, event& e)
  {
    is >> e.T >> e.NS >> e.pos.x >> e.pos.y >> e.sp >> e.ancestor;
    char delim; is >> delim;    // skip ';'
    return is;
  }

  
  std::ostream& operator<<(std::ostream& os, const Parameter& p)
  {
    os << p.L << ' ';
    os << p.L0 << ' ';
    os << p.Linc << ' ';
    os << p.Lg << ' ';
    os << p.ticks << ' ';
    os << p.mu << ' ';
    os << p.nu << ' ';
    os << p.s_nu << ' ';
    os << p.Psi << ' ';
    os << p.Phi << ' ';
    os << p.s_jc << ' ';
    os << p.jc_cutoff << ' ';
    os << p.s_disp << ' ';
    os << p.disp_cutoff << ' ';
    os << p.totalTicks << ' ';
    os << p.torus << ' ';
    os << p.nthreads << ' ';
    os << p.max_seq_num << ' ';
    os << '"' << p.filename.c_str() << "\" ";
    os << '"' << p.cont.c_str() << "\" ";
    os << p.log << ' ' << p.elapsed_time << ' ';
    return os;
  }


  std::istream& operator>>(std::istream& is, Parameter& p)
  {
    std::string version;
    is >> std::quoted(version);
    if (version != jc::Version) throw std::runtime_error("Version mismatch");
    is >> p.L;
    is >> p.L0;
    is >> p.Linc;
    is >> p.Lg;
    is >> p.ticks;
    is >> p.mu;
    is >> p.nu;
    is >> p.s_nu;
    is >> p.Psi;
    is >> p.Phi;
    is >> p.s_jc;
    is >> p.jc_cutoff;
    is >> p.s_disp;
    is >> p.disp_cutoff;
    is >> p.totalTicks;
    is >> p.torus;
    is >> p.nthreads;
    is >> p.max_seq_num;
    is >> std::quoted(p.filename);
    is >> std::quoted(p.cont);
    is >> p.log >> p.elapsed_time;
    return is;
  }


  template <typename T>
  std::ostream& operator<<(std::ostream& os, square_buffer<T> const& s)
  {
    for (size_t i = 0; i < s.n(); ++i) {
      for (size_t j = 0; j < s.n(); ++j) {
        os << s(j, i) << ' ';
      }
      os << ";\n";
    }
    return os;
  }
  
  
  template <typename T>
  std::istream& operator>>(std::istream& is, square_buffer<T>& s)
  {
    for (size_t i = 0; i < s.n(); ++i) {
      for (size_t j = 0; j < s.n(); ++j) {
        is >> s(j, i);
      }
      char delim;
      is >> delim;    // skip ';'
    }
    return is;
  }


  // converts CR LF to LF
  std::istream& save_getline(std::istream& is, std::string& str, char delim = '\n')
  {
    std::getline(is, str, delim);
    auto pos = str.rfind(0x0d);
    if (pos != str.npos) str.erase(pos, 1);
    return is;
  }


  std::istream& skip_to_label(std::istream& is, const char* label)
  {
    std::string str;
    while (save_getline(is, str)) {
      if (str == label) {
        return is;
      }
    }
    return is;
  }


  template <typename T>
  std::vector<T> ReadMat(std::istream& is)
  {
    std::vector<T> buf;
    std::string str;
    save_getline(is, str);      // skip assignment
    for (; is;) {
      save_getline(is, str, ';');
      if (str.empty() || *(str.end() - 1) == ']') break;
      std::stringstream ss(str);
      while (ss) {
        T val;
        ss >> val;
        if (ss) buf.push_back(val);
      }
    }
    save_getline(is, str);    // skip last ";" after closing bracket
    return buf;
  }


  Continuation::Continuation(const std::string& fname)
  {
    auto is = std::ifstream(fname);
    if (skip_to_label(is, cpp_skip_label_last_log)) {
      D_ = ReadMat<int>(is);
      R_ = ReadMat<int>(is);
      if (skip_to_label(is, cpp_skip_label_m)) {
        M_ = ReadMat<int>(is);
      }
      if (skip_to_label(is, cpp_skip_label_state)) {
        is >> param_;
      }
    }
  }


  MatLogger::MatLogger(const Parameter& param) : os_(param.filename)
  {
    if (!os_) throw std::invalid_argument((std::string("can't create data file ") + param.filename).c_str());

    auto today = std::time(nullptr);
    // spits out header
    os_ << "% " << param.filename.c_str();
    os_ << "\n% JanzenConnell result file.\n";
    os_ << "% " << jc::Version << '\n';
    os_ << "% Generated at " << std::ctime(&today);
    os_ << "\n%\n% Parameter set\n%\n";
    os_ << "L = " << param.L << "; % grid size\n";
    os_ << "L0 = " << param.L << "; % initial grid size\n";
    os_ << "Lg = " << param.L << "; % grid size growth\n";
    os_ << "ticks = " << param.ticks << "; % time steps [y]\n";
    os_ << "mu = " << param.mu << "; % 1/average lifetime [1/y]\n";
    os_ << "nu = " << param.nu << "; % speciation rate per death event'\n";
    os_ << "s_nu = " << param.s_nu << "; % sigma speciation radius\n";
    os_ << "Psi = " << param.Psi << "; % strength of JC effect\n";
    os_ << "Phi = " << param.Phi << "; % size of phylogenetic effect\n";
    os_ << "s_jc = " << param.s_jc << "; % sigma spacial distance\n";
    os_ << "jc_cutoff = " << param.jc_cutoff << "; % spacial distance cutoff\n";
    os_ << "s_disp = " << param.s_disp << "; % sigma dispersal\n";
    os_ << "disp_cutoff = " << param.s_disp << "; % dispersal cutoff\n";
    os_ << "totalTicks = " << param.totalTicks << "; % total ticks\n";
    os_ << "log = " << param.log << "; % Log interval for M, D and R [events]\n";
    os_ << "continuation = \"" << param.cont << "\"; % continuation simulation\n";
    os_ << "max_seq_number = " << param.max_seq_num << " % max events per timestep\n";
    os_ << EventHeader;
    os_ << "events = [];\n";
    os_ << "extinctions = [];\n";
    os_ << "speciations = [];\n";
    os_ << "\n%\n% Snapshot log of D, and R\n";
    os_ << "% the first and the last set in d simulation are always logged\n";
    os_ << "% use D{end} and R{end} to get the last records\n%\n";
    os_ << "D = {};    % cell array of D-matrices\n";
    os_ << "R = {};    % cell array of abundance histograms\n";
    os_ << "sT = [];   % vector of snapshot times\n";
    os_ << "% the last M matrix\n";
    os_ << "M = {};    % cell array of M-matrice\n";
    os_ << "\n\n%!!!!!!!!!!!!! DO NOT EDIT AFTER THIS LINE !!!!!!!!!!!!!\n\n";
    os_ << cpp_skip_label_read_back << std::endl;
  }


  MatLogger::~MatLogger()
  {
  }


  void MatLogger::logEvents(std::vector<event> const& events)
  {
    os_ << cpp_skip_label_event_log << '\n';
    os_ << "\nevents = [\n";
    for (const auto& e : events) { os_ << e << '\n'; }
    os_ << "];\n";
    os_ << "if (length(events) > 0)\n";
    os_ << "    extinctions = events(events(:, 6) == -1, :);\n";
    os_ << "    speciations = events(events(:, 6) >= 0, :);\n";
    os_ << "end;" << std::endl;
  }


  void MatLogger::logSnapshot(int64_t T, const GDM& D, std::vector<int> const& R)
  {
    os_ << "D{length(D)+1} = [\n";
    os_ << D.Dij() << "];" << std::endl;
    os_ << "R{length(R)+1} = [\n";
    for (size_t i = 0; i < R.size(); ++i) { os_ << R[i] << ' '; }
    os_ << ";\n];" << std::endl;
    os_ << "sT = [sT; " << T << "];" << std::endl;
  }


  void MatLogger::logLastSnapshot(int64_t T, square_buffer<int> const& M, const GDM& D, std::vector<int> const& R)
  {
    os_ << cpp_skip_label_last_log << '\n';
    logSnapshot(T, D, R);
    os_ << cpp_skip_label_m << '\n';
    os_ << "M{length(M)+1} = [\n";
    os_ << M << "];" << std::endl;
  }
    

  void MatLogger::logState(const Parameter& param)
  {
    os_ << "elapsedTime = " << param.elapsed_time << ";\n\n";
    os_ << "\n%{\n";
    os_ << cpp_skip_label_state << '\n';
    os_ << '"' << jc::Version << "\"\n";
    os_ << param << '\n';
    os_ << "%}" << std::endl;
  }

} // namespace jc

