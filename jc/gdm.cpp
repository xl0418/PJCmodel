#include <cassert>
#include <utility>
#include <numeric>
#include <algorithm>
#include <random>
#include "gdm.h"
#include "jc.h"


namespace jc {
  

  // exp(-Phi^2 * Dij^2)
  std::vector<real_t> make_dij_exp_lookup(const Parameter& param)
  {
    const auto Phi = param.Phi;
    if (Phi <= 0.0) {
      // 'infinite'
      return std::vector<real_t>({ 1 });
    }
    // Quote from cppreference.com: 
    // "For IEEE-compatible type double, overflow is guaranteed if 709.8 < arg, and underflow is guaranteed if arg < -708.4"
    // exp(-Phi^2 * x^2) == 0 (underflow) => maxDij^2 > 708.4 / Phi^2
    // 
    const auto ss = param.Phi * param.Phi;
    const auto eps = 10e-10;  // std::numeric_limits<double>::epsilon();
    std::vector<real_t> lookup;
    size_t dij = 0;
    for (;; ++dij) {
      const auto expdij = std::exp(-ss * double(dij) * dij);
      if ((expdij < eps) || (std::fpclassify(static_cast<real_t>(expdij)) == FP_SUBNORMAL)) {
        break;
      }
      lookup.push_back(static_cast<real_t>(expdij));
    }
    lookup.push_back(0);  // guard
    return lookup;
  }


  GDM::GDM(const Parameter& param) :
    Dij_(2ull, 64ull, 0, true),
    expDij_(2ull, 64ull, true),
    expdij_lookup_(make_dij_exp_lookup(param)),
    reng_(rndutils::make_random_engine<>()),
    s_nu_(param.s_nu),
    Psi_(param.Psi)
  {
    update(param.L, { real_t(param.L * param.L - 1), real_t(1) }, 1);
  }


  GDM::GDM(const struct Parameter& param, const std::vector<real_t>& Rf, square_buffer<int>&& Dij) :
    Dij_(std::move(Dij)),
    expdij_lookup_(make_dij_exp_lookup(param)),
    reng_(rndutils::make_random_engine<>()),
    s_nu_(param.s_nu),
    Psi_(param.Psi)
  {
    update(param.L, Rf, 0);  // refresh
  }

#ifdef JC_AVX2

  void GDM::update_simd(const std::vector<real_t>& Rf, int incr)
  {
    const auto n = static_cast<int>(Dij_.n());
    const auto maxdij = static_cast<int>(expdij_lookup_.size() - 1);    // expdij_lookup_[maxdij] maps to zero
    const __m256i incr256 = _mm256_set1_epi32(incr);
    const __m256i maxdij256 = _mm256_set1_epi32(maxdij);

    tbb::parallel_for(0, n, [&, Rf = Rf.data(), lookup = expdij_lookup_.data()](int i) {
      auto Di = Dij_.row(i);
      auto Ei = expDij_.row(i);
      auto Pimp = 0.f;
      __m256 Pimp256 = _mm256_setzero_ps();
      auto inner = [&](int j, int j1) {
        const auto chunks = (j1 - j) >> 3;
        for (auto c = 0; c < chunks; ++c, j += 8) {
          const __m256i dij = _mm256_add_epi32(_mm256_loadu_si256((__m256i*)&Di[j]), incr256);
          const __m256 rfj = _mm256_loadu_ps(&Rf[j]);
          const __m256i clamped_dij = _mm256_min_epi32(dij, maxdij256);
          const __m256 Eij = _mm256_i32gather_ps(lookup, clamped_dij, 4);
          _mm256_storeu_si256((__m256i*)&Di[j], dij);
          _mm256_storeu_ps(&Ei[j], Eij);
          Pimp256 = _mm256_fmadd_ps(rfj, Eij, Pimp256);
        }
        // scalar tail
        for (; j < j1; ++j) {
          const auto dij = std::min((Di[j] += incr), maxdij);
          const auto e = Ei[j] = lookup[dij];
          Pimp += Rf[j] * e;
        }
      };
      inner(0, i);
      inner(i + 1, n);
      Pimp_[i] = Pimp + simd::reduce_add(Pimp256);
    });
  }

#else

  void GDM::update_simd(const std::vector<real_t>& Rf, int incr)
  {
    const auto n = static_cast<int>(Dij_.n());
    const auto maxdij = static_cast<int>(expdij_lookup_.size() - 1);    // expdij_lookup_[maxdij] maps to zero
    tbb::parallel_for(0, n, [&, Rf = Rf.data()](int i) {
      auto Di = Dij_.row(i);
      auto Ei = expDij_.row(i);
      auto Pimp = real_t(0);
      auto inner = [&](int j, int j1) {
        for (; j < j1; ++j) {
          const auto dij = std::min((Di[j] += incr), maxdij);
          const auto e = Ei[j] = expdij_lookup_[dij];
          Pimp += Rf[j] * e;
        }
      };
      inner(0, i);
      inner(i + 1, n);
      Pimp_[i] = Pimp;
    });
  }

#endif


  void GDM::update(int L, const std::vector<real_t>& Rf, int incr)
  {
    const auto n = static_cast<int>(Dij_.n());
    expDij_.resize(n);
    Pimp_.resize(n);
    update_simd(Rf, incr);
    const auto scale = real_t(Psi_ / (double(L) * L));
    for (int i = 0; i < n; ++i) {
      Dij_(i, i) = 0;
      expDij_(i, i) = real_t(1);
      Pimp_[i] = std::clamp(real_t(1) - scale * Pimp_[i], real_t(0), real_t(1));  // capture freak numeric issues in sum above 
    }
    // only used by ImplicitSampler but not very costly
    PimpCDF_.mutate(Pimp_.cbegin(), Pimp_.cend());
  }


  void GDM::extinction(int speciesId)
  {
    auto bv = [](auto& b, auto w, auto h) { return make_block_view(b, w, h); };
    const size_t n = Dij_.n();
    const size_t s = speciesId;
    const size_t r = n - speciesId - 1;
    copy(bv(Dij_, r, s).shift(s, 0), bv(Dij_, r, s).shift(s + 1, 0));      // top right
    copy(bv(Dij_, s, r).shift(0, s), bv(Dij_, s, r).shift(0, s + 1));      // bottom left
    copy(bv(Dij_, r, r).shift(s, s), bv(Dij_, r, r).shift(s + 1, s + 1));  // bottom right
    Dij_.resize(n - 1);
  }


  int GDM::specification(const position& pos, int nsp, square_buffer<int>& M)
  {
    const auto ancestorId = M(pos.x, pos.y);
    M(pos.x, pos.y) = nsp;    // this one is guarantied to speciate
    const auto torus = M.torus();
    auto nudist = std::normal_distribution<>(0.0, s_nu_);
    const auto kd = std::abs(nudist(reng_)); 
    const auto k = std::min(static_cast<int>(M.n())/4 - 2, static_cast<int>(kd));     // kernel size
    const auto kk = k * k;
    int nn = 1;
    for (int y = -k; y <= k; ++y) {
      const auto wy = torus.wrap(pos.y + y);
      const auto ddy = torus.dist2(wy, pos.y);
      const auto* Mj = M.row(wy);
      for (int x = -k; x <= k; ++x) {
        const auto wx = torus.wrap(pos.x + x);
        const auto dd = torus.dist2(wx, pos.x) + ddy;
        if ((dd && (dd <= kk)) && (Mj[wx] == ancestorId)) {
          M(wx, wy) = nsp;
          ++nn;
        }
      }
    }
    // adjust D matrix
    auto bv = [](auto& b, auto w, auto h) { return make_block_view(b, w, h); };
    const size_t n = Dij_.n();
    if (Dij_.stride() < n + 1) {
      auto tmp = square_buffer<int>(n + 64);          // over-subscription
      parallel_copy(bv(tmp, n, n), bv(Dij_, n, n));   // copy matrix
      Dij_.swap(tmp);
    }
    Dij_.resize(n + 1);   // no reallocation
    copy(bv(Dij_, n, 1).shift(0, n), bv(Dij_, n, 1).shift(0, ancestorId));  // duplicate row into last row
    copy(bv(Dij_, 1, n).shift(n, 0), bv(Dij_, 1, n).shift(ancestorId, 0));  // duplicate column into last column
    Dij_(n, n) = 0;                                                         // distance to itself <- 0
    Dij_(n, ancestorId) = Dij_(ancestorId, n) = 1;                          // distance to ancestor <- 1
    return nn;
  }


} // namespace jc
