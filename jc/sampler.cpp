#include <cassert>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <algorithm>
#include <atomic>
#include <execution>
#include "rndutils.hpp"
#include "sampler.h"
#include "jc_config.hpp"
#include "jc_simd.h"


namespace jc {


  // samples the index in the CDF given in [first, last)
  // handles all-zero case as uniform probability
  template <typename T, typename IT, typename RENG>
  inline int sample_cdf(T sum, IT first, IT last, RENG& reng)  noexcept
  {
    if (sum) {
      const auto p = sum * rndutils::uniform01<real_t>(reng);
      return static_cast<int>(std::distance(first, std::lower_bound(first, last, p)));
    }
    return std::uniform_int_distribution<int>(0, static_cast<int>(std::distance(first, last)) - 1)(reng);
  }


  class NeutralSampler : public Sampler
  {
  public:
    NeutralSampler(const Parameter&) : Sampler() {};

    int draw(const position& pos,
             const state_t& state,
             const kernels_t&,
             reng_t& reng) override
    {
      // direct sampling
      auto pdist = std::uniform_int_distribution<int>(0, static_cast<int>(state.M.n()) - 1);
      auto p = position{ pdist(reng), pdist(reng) };
      while (p == pos) {
        p = position{ pdist(reng), pdist(reng) };
      }
      return state.M(p.x, p.y);
    }
  };


  class NeutralDispSampler : public Sampler
  {
  public:
    NeutralDispSampler(const Parameter& param) : Sampler() {};

    int draw(const position& pos,
             const state_t& state,
             const kernels_t& kernels,
             reng_t& reng) override
    {
      const auto& M = state.M;
      const auto torus = M.torus();
      const auto k = static_cast<int>(kernels.Kdisp.K().n());  // kernel size
      const auto i = kernels.Kdisp.cdf(reng);
      const auto kx = i % k - (k >> 1);     // offset to kernel center
      const auto ky = i / k - (k >> 1);     // offset to kernel center
      return M(torus.wrap(pos.x + kx), torus.wrap(pos.y + ky));
    }
  };


  class ImplicitSampler : public Sampler
  {
  public:
    ImplicitSampler(const Parameter& param) : Sampler() {}

    int draw(const position& /* pos */,
             const state_t& state, 
             const kernels_t& /* kernels */,
             reng_t& reng) override
    {
      return state.D.PimpCDF()(reng);
    }
  };


  class ImplicitDispSampler : public Sampler
  {
  public:
    explicit ImplicitDispSampler(const Parameter& param) :
      Sampler(),
      Pi_(2ull * param.disp_cutoff + 1)
    {}

    int draw(const position& pos,
             const state_t& state,
             const kernels_t& kernels,
             reng_t& reng) override
    {
      const auto& Pimp = state.D.Pimp();
      const auto torus = state.M.torus();
      const auto& Kdisp = kernels.Kdisp.K();
      const auto k = static_cast<int>(Kdisp.n());  // kernel size
      const auto pc = position{ pos.x - (k >> 1), pos.y - (k >> 1) };   // shift by center of kernel
      // build (linear) cdf on the fly
      auto sum = real_t(0);
      for (int y = 0; y < k; ++y) {
        const auto* Mj = state.M.row(torus.wrap(pc.y + y));
        auto* __restrict Pi = Pi_.row(y);
        const auto* __restrict Kd = Kdisp.row(y);
        for (int x = 0; x < k; ++x) {
          const auto sp = Mj[torus.wrap(pc.x + x)];
          Pi[x] = (sum += Pimp[sp] * Kd[x]);
        }
      }
      const auto i = sample_cdf(sum, Pi_.cbegin(), Pi_.cend(), reng);
      const auto kx = i % k;      // linear to x
      const auto ky = i / k;      // linear to y
      return state.M(torus.wrap(pc.x + kx), torus.wrap(pc.y + ky));
    }

  private:
    square_buffer<real_t> Pi_;
  };


  class ExplicitSampler : public Sampler
  {
  public:
    explicit ExplicitSampler(const Parameter& param) :
      Sampler(),
      K_(2ll * param.jc_cutoff + 1),
      Ksp_((2ll * param.jc_cutoff + 1) * (2ll * param.jc_cutoff + 1)),
      Kp_((2ll * param.jc_cutoff + 1)* (2ll * param.jc_cutoff + 1)),
      KexpDij_(2ll * param.jc_cutoff + 1)
    {}

    int draw(const position& pos,
             const state_t& state,
             const kernels_t& kernels,
             reng_t& reng) override
    {
      p_explicit(pos, state, kernels);
      // build cdf
      auto sum = simd::inclusive_scan_mul(Pexpl_.data(), Pexpl_.data() + Pexpl_.size(), state.Rf.data(), Pexpl_.data());
      return sample_cdf(sum, Pexpl_.cbegin(), Pexpl_.cend(), reng);
    }

  protected:
    //                    sum_grid( exp(-Phi^2 * Dij^2) * exp(d^2/(2 s_jc^2) )
    // P_[i] <- 1 - Psi * ---------------------------------------------------------
    //                                sum_grid( exp(d^2/(2 s_jc^2) )
    using gather_K = std::integral_constant<bool, true>;

    void p_explicit(const position& pos,
                    const state_t& state,
                    const kernels_t& kernels);

    std::vector<real_t> Pexpl_;       // p_explicit for each species in system

  private:
    square_buffer<int> K_;
    std::vector<int> Ksp_;            // species inside jc kernel
    std::vector<int8_t> Ksp_map_;     // Ksp_[Ksp_map_[sp]] == sp
    std::vector<real_t> Kp_;          // p_explicit per species inside jc kernel
    square_buffer<real_t> KexpDij_;   // expDij matrix of species in jc kernel
  };


  //                    sum_grid( exp(-Phi^2 * Dij^2) * exp(d^2/(2 s_jc^2) )
  // P_[i] <- 1 - Psi * ---------------------------------------------------------
  //                                sum_grid( exp(d^2/(2 s_jc^2) )
  void ExplicitSampler::p_explicit(const position& pos,
                                   const state_t& state,
                                   const kernels_t& kernels)
  {
    const auto torus = state.M.torus();
    const auto& expDij = state.D.expDij();
    const auto& Kjc = kernels.Kjc.K();
    const auto k = static_cast<int>(Kjc.n());  // kernel size
    const auto pc = position{ pos.x - (k >> 1), pos.y - (k >> 1) };   // shift by center of kernel
    const auto NS = static_cast<int>(state.Rf.size());         // number of species

    Ksp_map_.assign(NS, 1);
    int ks = 0;
    auto* k_it = K_.data();
    auto* jc_it = Kjc.data();
    for (int y = 0; y < k; ++y) {
      const auto* Mj = state.M.row(torus.wrap(pc.y + y));
      for (int x = 0; x < k; ++x, ++k_it, ++jc_it) {
        const auto sp = Mj[torus.wrap(pc.x + x)];
        *k_it = sp;
        if (Ksp_map_[sp] && *jc_it) {
          Ksp_[ks] = sp;
          Ksp_map_[sp] = 0;
          ks++;
        }
      }
    }
    
    // species outside jc kernel have p_explicit == p_implicit
    Pexpl_ = state.D.Pimp();

    for (auto ki = 0; ki < ks; ++ki) {
      const auto* edi = expDij.row(Ksp_[ki]);
      const auto* sp_it = K_.data();
      const auto* jc_it = Kjc.data();
      auto Kpi = real_t(0);
      int i = 0;
#ifdef JC_AVX2
      auto chunks = (k * k) >> 3;
      __m256 Kpi256 = _mm256_setzero_ps();
      for (int c = 0; c < chunks; ++c, i += 8, sp_it += 8, jc_it += 8) {
        const __m256i spv = _mm256_loadu_si256((__m256i*)(sp_it));
        const __m256 jcv = _mm256_loadu_ps(jc_it);
        const __m256 ediv = _mm256_i32gather_ps(edi, spv, 4);
        Kpi256 = _mm256_fmadd_ps(ediv, jcv, Kpi256);
      }
      Kpi = simd::reduce_add(Kpi256);
#endif
      // scalar (tail)
      for (;  i < k * k; ++i, ++sp_it, ++jc_it) {
        Kpi += edi[*sp_it] * *jc_it;
      }
      Pexpl_[Ksp_[ki]] = 1.0f - Kpi;
    }
  }


  class ExplicitDispSampler : public ExplicitSampler
  {
  public:
    explicit ExplicitDispSampler(const Parameter& param) :
      ExplicitSampler(param),
      cdf_((2ll * param.disp_cutoff + 1) * (2ll * param.disp_cutoff + 1))
    {}

    int draw(const position& pos,
             const state_t& state,
             const kernels_t& kernels,
             reng_t& reng) override
    {
      using std::get;
      p_explicit(pos, state, kernels);
      const auto torus = state.M.torus();
      const auto& Kdisp = kernels.Kdisp.K();
      const auto k = static_cast<int>(Kdisp.n());  // dispersal kernel size
      const auto pc = position{ pos.x - (k >> 1), pos.y - (k >> 1) };   // shift by center of kernel

      // build (linear) cdf
      auto cdf = cdf_.data();
      auto sum = real_t(0);
#ifdef JC_AVX2
      static const __m256i him = _mm256_set1_epi32(7);
      __m256 sumv = _mm256_setzero_ps();
#endif
      const auto xr = torus.kernel_ranges(pos.x, k >> 1);
      for (int y = 0; y < k; ++y) {
        const auto* My = state.M.row(torus.wrap(pc.y + y));
        const auto* Kd = Kdisp.row(y);
        int x0 = get<0>(xr).begin;
#ifdef JC_AVX2
        const auto chunks = (get<0>(xr).end - x0) >> 3;
        for (auto c = 0; c < chunks; ++c, x0 += 8, cdf += 8, Kd += 8) {
          __m256i spv = _mm256_loadu_si256((__m256i*)&My[x0]);
          __m256 Kdv = _mm256_loadu_ps(Kd);
          __m256 p = _mm256_i32gather_ps(Pexpl_.data(), spv, 4);
          p = _mm256_mul_ps(p, Kdv);
          sumv = _mm256_add_ps(sumv, simd::inclusive_scan_ps(p));
          _mm256_storeu_ps(cdf, sumv);
          sumv = _mm256_permutevar8x32_ps(sumv, him);
        }
        if (chunks) sum = *(cdf - 1);
#endif
        // scalar (tail)
        for (; x0 < get<0>(xr).end; ++x0, ++cdf, ++Kd) {
          *cdf = (sum += Pexpl_[My[x0]] * *Kd);
        }
        // second range is the smaller one - and rarely non-empty
        for (auto x1 = get<1>(xr).begin; x1 < get<1>(xr).end; ++x1, ++cdf, ++Kd) {
          *cdf = (sum += Pexpl_[My[x1]] * *Kd);
        }
        int dummy = 0;
      }
      auto i = sample_cdf(sum, cdf_.cbegin(), cdf_.cend(), reng);
      const auto kx = i % k;
      const auto ky = i / k;
      return state.M(torus.wrap(pc.x + kx), torus.wrap(pc.y + ky));
    }

  private:
    std::vector<real_t> cdf_;     // individual probability dispersal kernel
  };


  // Psi   s_disp   s_jc    Sampler
  // -------------------------------------------
  // <= 0  <= 0     n.a.    NeutralSampler
  // <= 0  > 0      n.a.    NeutralDispSampler
  // > 0   <= 0     <= 0    ImplicitSampler
  // > 0   > 0      <= 0    ImplicitDispSampler
  // > 0   <= 0     > 0     ExplicitSampler
  // > 0   > 0      > 0     ExplicitDispSampler
  //

  Sampler* Sampler::create(const Parameter& param)
  {
    if (param.Psi <= 0) {
      if (param.s_disp <= 0) {
        return new NeutralSampler(param);
      }
      return new NeutralDispSampler(param);
    }
    if (param.s_jc <= 0) {
      if (param.s_disp <= 0) {
        return new ImplicitSampler(param);
      }
      else {
        return new ImplicitDispSampler(param);
      }
    }
    if (param.s_disp <= 0) {
      return new ExplicitSampler(param);
    }
    return new ExplicitDispSampler(param);
  }


  const char* Sampler::get_name(const Parameter& param)
  {
    if (param.Psi <= 0) {
      if (param.s_disp <= 0) {
        return "NeutralSampler";
      }
      return "NeutralDispSampler";
    }
    if (param.s_jc <= 0) {
      if (param.s_disp <= 0) {
        return "ImplicitSampler";
      }
      else {
        return "ImplicitDispSampler";
      }
    }
    if (param.s_disp <= 0) {
      return "ExplicitSampler";
    }
    return "ExplicitDispSampler";
  }

} // namespace jc
