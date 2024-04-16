#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "jc_config.hpp"
#include "game_watches.hpp"


constexpr int N = 5000;

int main()
{
  std::vector<float> v(N);
  std::iota(v.begin(), v.end(), 1.f);

  auto cdf0 = std::vector<float>(N);
  auto cdf1 = std::vector<float>(N);
  auto cdf2 = std::vector<float>(N);

  auto watch0 = game_watches::stop_watch{};
  auto watch1 = game_watches::stop_watch{};
  auto watch2 = game_watches::stop_watch{};

  for (int i = 0; i < 100'000; ++i) {
    watch0.start();
    auto sum0 = *(std::inclusive_scan(v.cbegin(), v.cend(), cdf0.begin()) - 1);
    watch0.stop();
    watch1.start();
    auto sum1 = jc::simd::inclusive_scan_256(v.data(), v.data() + N, cdf1.data());
    watch1.stop();
    watch2.start();
    auto sum2 = sum1; // jc::simd::inclusive_scan_512(v.data(), v.data() + N, cdf2.data());
    watch2.stop();
    if ((sum0 != sum1) || (sum0 != sum2)) std::cout << '.';
  }
  std::cout << watch0.elapsed<std::chrono::milliseconds>().count() << '\n';
  std::cout << watch1.elapsed<std::chrono::milliseconds>().count() << '\n';
  std::cout << watch2.elapsed<std::chrono::milliseconds>().count() << '\n';
  return 0;
}
