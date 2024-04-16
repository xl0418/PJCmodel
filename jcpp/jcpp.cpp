#include <cassert>
#include <iostream>
#include <sstream>
#include <string_view>
#include <set>
#include <deque>
#include <array>
#include <limits>
#include "iomatlab.h"
#include "jc_config.hpp"
#include "cmd_line.h"

#if __has_include(<filesystem>)
	#include <filesystem>
	namespace fs = std::filesystem;
#else
	#include <experimental/filesystem>
	namespace fs = std::experimental::filesystem;
#endif


// parameter
constexpr int64_t C = 64 * 1024 * 1024;   // chunk size (64MB)
bool verbose = false;
fs::path output;


struct Ltable
{
  int64_t t;    // branching time
  int64_t s;    // parent species
  int64_t a;    // daughter species
  int64_t e;    // extinction time
};


#pragma pack(push, 1)
template <typename T>
struct TLtable
{
  T t;    // branching time
  T s;    // parent species
  T a;    // daughter species
  T e;    // extinction time
};
#pragma pack(pop)



template <typename T>
auto strtoint(const char* it, const char** iit) -> std::enable_if_t<std::is_signed_v<T>, T>
{
  T x = T(0);
  while (*it == ' ') ++it;
  bool neg = *it == '-';
  if (neg) ++it;
  while (*it >= '0' && *it <= '9') {
    x = (x * T(10)) + (T(*it) - '0');
    ++it;
  }
  *iit = it;
  return neg ? -x : x;
}


template <typename T>
auto strtoint(const char* it, const char** iit) -> std::enable_if_t<std::is_unsigned_v<T>, T>
{
  T x = T(0);
  while (*it == ' ') ++it;
  while (*it >= '0' && *it <= '9') {
    x = (x * T(10)) + (T(*it) - '0');
    ++it;
  }
  *iit = it;
  return x;
}


struct table_t
{
  std::array<int64_t, 4> header = { 0,-1, 1,-1 };    // holds info about 0,1
  std::deque<Ltable> deque;
  std::vector<int64_t> sp;
};


table_t read_ltable(const fs::path& path)
{
  int64_t totalTicks = 0;
  int64_t totalEvents = 0;
  {
    jc::Parameter param;
    std::ifstream is(path);
    is.seekg(-1024, is.end);
    if (jc::skip_to_label(is, jc::cpp_skip_label_state)) {
      is >> param;
    }
    else {
      throw std::runtime_error("input file corrupted");
    }
    if (!param.cont.empty()) throw "continuation not supported";
    totalTicks = param.totalTicks;
    if (verbose) std::cout << "Ticks: " << totalTicks << '\n';
    totalTicks *= jc::max_sequence_number;    // adjust for events per tick
  }
  table_t table{};
  std::deque<Ltable>& events = table.deque;
  std::ifstream is(path, std::ios::binary);
  jc::skip_to_label(is, "events = [");    // relatively fast
  int64_t first = is.tellg();
  is.seekg(0, is.end);
  const int64_t last = is.tellg();
  auto buffer = std::unique_ptr<char[]>(new char[C]);
  auto sp = std::vector<int64_t>{ 0,1 };    // starting species
  int64_t nextsp = 2;                       // next specieated species

  if (verbose) std::cout << "parsing events ";
  for (;;) {
    if (verbose) std::cout << '.';
    auto chunk = std::min(last - first, C);
    is.seekg(first);
    is.read(buffer.get(), chunk);
    std::string_view sv(buffer.get(), chunk);
    auto lpos = sv.find_last_of(';');
    if (lpos == sv.npos || lpos == 0) break;
    sv.remove_suffix(sv.size() - (lpos + 1));
    while (!sv.empty()) {
      ++totalEvents;
      auto lbeg = sv.find(' ');
      auto lend = sv.find(';');
      if (lend <= lbeg) break;
      auto line = sv.substr(lbeg + 2, lend - lbeg);
      const char* it = nullptr;
      auto t = strtoint<int64_t>(line.data(), &it);
      strtoint<int64_t>(it, &it);
      strtoint<int64_t>(it, &it);
      strtoint<int64_t>(it, &it);
      auto s = strtoint<int64_t>(it, &it);
      auto a = strtoint<int64_t>(it, &it);
      if (a == -1) {
        // extinction
        auto ext_sp = sp[s];
        auto it = std::lower_bound(events.begin(), events.end(), ext_sp, [=](const auto& a, auto b) { return a.s < b; });
        if ((it != events.end()) && (it->s == ext_sp)) {
          it->e = totalTicks - t;   // record extinction time
        }
        // special handling for initial species
        // record extinction time
        if (s == 0 && table.header[1] == -1) {
          table.header[1] = totalTicks - t;
        }
        else if (s == 1 && table.header[3] == -1) {
          table.header[3] = totalTicks - t;
        }
        sp.erase(sp.begin() + s);
      }
      else {
        sp.push_back(nextsp);
        events.push_back({ totalTicks - t, nextsp, sp[a], -1 });
        ++nextsp;
      }
      sv.remove_prefix(lend + 1);
    }
    first += lpos + 1;
  }
  if (verbose) {
    std::cout << "\nevents: " << totalEvents << '\n';
    std::cout << "L table: " << events.size() << '\n';
    std::cout << "tips: " << sp.size() << '\n';
  }
  table.sp = std::move(sp);
  return table;
}


template <typename T>
void write_full_ltable(const std::deque<Ltable>& L)
{
  std::string ext;
  if constexpr (std::is_same_v<T, int32_t>) ext = ".i32.bin";
  if constexpr (std::is_same_v<T, int64_t>) ext = ".i64.bin";
  if constexpr (std::is_same_v<T, double>) ext = ".f64.bin";
  auto os = std::ofstream(output.replace_extension(ext), std::ios::out | std::ios::binary);
  if (!os.is_open()) throw "can't create output file";
  if (verbose) std::cout << "writing full L table ";
  std::vector<TLtable<T>> buf(C / sizeof(TLtable<T>));
  auto it = buf.begin();
  for (size_t i = 0; i < L.size(); ++i, ++it) {
    if (it == buf.end()) {
      os.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(TLtable<T>));
      it = buf.begin();
      if (verbose) std::cout << '.';
    }
    // swap s with a
    *it = { static_cast<T>(L[i].t), static_cast<T>(L[i].a), static_cast<T>(L[i].s), static_cast<T>(L[i].e) };
  }
  const auto bytes = std::distance(buf.begin(), it) * sizeof(TLtable<T>);
  os.write((const char*)buf.data(), bytes);
  if (verbose) std::cout << ".\n";
};


struct lt_cmp
{
  template <typename LT>
  bool operator()(const LT* a, const LT* b) const noexcept { return a->t < b->t; }
};


int64_t index_of(int64_t sp, const table_t& table)
{
  if (sp < 0) {
    return sp;
  }
  auto it = std::lower_bound(table.sp.begin(), table.sp.end(), sp, [=](auto a, auto b) { return a < b; });
  assert(it != table.sp.end());
  return std::distance(table.sp.begin(), it);
}


void write_csv(table_t& table)
{
  if (verbose) std::cout << "pruning tree";
  auto os = std::ofstream(output.replace_extension(".csv"));
  if (!os.is_open()) throw "can't create csv file";

  // handle species 0 & 1
  auto& L = table.deque;
  if (table.header[1] != -1) {
    auto it = std::find_if(L.rbegin(), L.rend(), [=](auto& a) { return a.t > table.header[1]; });
    std::for_each(it, L.rend(), [](auto& a) { if (a.a == 0) a.a = -1; });
  }
  if (table.header[3] != -1) {
    auto it = std::find_if(L.rbegin(), L.rend(), [=](auto& a) { return a.t > table.header[3]; });
    std::for_each(it, L.rend(), [](auto& a) { if (a.a == 1) a.a = -2; });
  }

  // collect all species that are in the lineages of present species
  std::set<Ltable*, lt_cmp> lineage;
  for (auto& lt : L) {
    if (lt.e == -1) {
      auto present = lineage.insert(&lt).first;
      auto a = (*present)->a;
      for (auto it = L.rbegin();;) {
        it = std::find_if(it, L.rend(), [a = a](auto plt) { return plt.s == a; });
        if (it == L.rend() || !lineage.insert(&*it).second) break;
        a = it->a;
        ++it;
      }
    }
  }

  std::vector<Ltable> tmp;
  for (auto plt : lineage) tmp.push_back(*plt);
  for (auto it = tmp.begin();;) {
    it = std::find_if(it, tmp.end(), [](const auto& lt) { return lt.e != -1; });
    if (it == tmp.end()) break;
    // replace earlier occurrence of extinct species as ancestor with its ancestor
    auto s = it->s;
    auto a = it->a;
    for (auto rit = decltype(tmp)::reverse_iterator{it}; rit != tmp.rend(); ++rit) {
      if (rit->a == s) rit->a = a;
    }
    *it = { -1,-1,-1,-1 };    // mark as handled
  }
  if (verbose) std::cout << "\nwriting pruned L table";
  os << table.header[0] << ',';
  os << double(table.header[1]) / jc::max_sequence_number << ',';
  os << table.header[2] << ',';
  os << double(table.header[3]) / jc::max_sequence_number << '\n';
  for (auto it = tmp.rbegin(); it != tmp.rend(); ++it) {
    if (it->t != -1) {
      auto s = index_of(it->s, table);
      auto a = index_of(it->a, table);
      // swap s with a
      os << double(it->t) / jc::max_sequence_number << ',' << a << ',' << s << ',' << -1 << '\n';
    }
  }
  if (verbose) std::cout << std::endl;
}


const char* usage = R"(
jcpp [Options] format file output
  file            input jc result file
options:
  --help          prints this message and exits
  --version       prints version information and exits
  --verbose, -v   verbose output
)";


int main(int argc, const char** argv)
{
  try {
    cmd::cmd_line_parser clp(argc, argv);
    if (clp.flag("--version"))
    {
      std::cout << jc::Version << '\n';
      std::cout << jc::ReportBugs << std::endl;
      return 0;
    }
    if (clp.flag("--help")) {
      std::cout << usage << '\n';
	  return 0;
    }
    verbose = clp.flag("-v") || clp.flag("--verbose");
    auto path = clp.required<fs::path>("file");
    if (!fs::exists(path)) throw cmd::parse_error((path.string() + " doesn't exists").c_str());

    output = path;
    output.replace_extension(".csv");
    auto table = read_ltable(path);
    auto& events = table.deque;
    write_csv(table);
    if (verbose) std::cout << "cleaning up";
    events.clear();
    events.shrink_to_fit();
    if (verbose) std::cout << std::endl;
    return 0;
  }
  catch (const cmd::parse_error & err) {
    std::cerr << err.what() << '\n';
    std::cout << usage;
  }
  catch (const std::exception & err) {
    std::cerr << err.what() << '\n';
  }
  catch (const char* err) {
    std::cerr << err << '\n';
  }
  return -1;
}
