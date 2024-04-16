#ifndef JC_SIMD_H_INCLUDED
#define JC_SIMD_H_INCLUDED

#ifndef JC_JC_CONFIG_HPP_INCLUDED
# error jc_simd.h is intended to be included by or after jc_confg.hpp
#endif


# include <immintrin.h>

namespace jc {

  namespace simd {

#ifdef JC_SIMD

    static_assert(std::is_same_v<real_t, float>, "AVX requires real_t to be float");


    inline __m128 reduce_add_ps(__m256 sv)
    {
      const __m128 lo4 = _mm256_castps256_ps128(sv);
      const __m128 hi4 = _mm256_extractf128_ps(sv, 1);
      const __m128 s4 = _mm_add_ps(lo4, hi4);
      const __m128 hi2 = _mm_movehl_ps(s4, s4);
      const __m128 s2 = _mm_add_ps(s4, hi2);
      const __m128 hi = _mm_shuffle_ps(s2, s2, 0x1);
      const __m128 s = _mm_add_ps(s2, hi);
      return s;
    }


    inline float reduce_add(__m256 sv)
    {
      return _mm_cvtss_f32(reduce_add_ps(sv));
    }


    // cross-lane shift by k elements
    inline __m256 sll1_ps(__m256 x)
    {
#ifdef JC_AVX512
      __m256i xi = _mm256_castps_si256(x);
      return _mm256_castsi256_ps(_mm256_maskz_alignr_epi32(~1, xi, xi, 0b111));
#else
      static const __m256i idx = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 0);
      auto xx = _mm256_permutevar8x32_ps(x, idx);
      return _mm256_castsi256_ps(_mm256_insert_epi32(_mm256_castps_si256(xx), 0, 0));
#endif
    }


    inline __m256 inclusive_scan_ps_old(__m256 val)
    {
      __m256 s = sll1_ps(val);
      __m256 c = _mm256_add_ps(val, s);
      s = sll1_ps(s); c = _mm256_add_ps(c, s);
      s = sll1_ps(s); c = _mm256_add_ps(c, s);
      __m256 x = _mm256_permute2f128_ps(c, c, 0);
#ifdef JC_AVX512
      return _mm256_mask_add_ps(c, 0b11110000, x, c);
#else
      __m128 lo = _mm256_extractf128_ps(x, 1);
      x = _mm256_add_ps(x, c);
      return _mm256_insertf128_ps(x, lo, 0);
#endif
    }


    inline __m256 inclusive_scan_ps(__m256 val)
    {
      __m256 s = sll1_ps(val);
      __m256 c = _mm256_add_ps(val, s);
      s = sll1_ps(s); c = _mm256_add_ps(c, s);
      s = sll1_ps(s); c = _mm256_add_ps(c, s);
      __m256 x = _mm256_permute2f128_ps(c, c, 0);
#ifdef JC_AVX512
      return _mm256_mask_add_ps(c, 0b11110000, x, c);
#else
      __m128 lo = _mm256_extractf128_ps(x, 1);
      x = _mm256_add_ps(x, c);
      return _mm256_insertf128_ps(x, lo, 0);
#endif
    }


    // returns sum
    inline float inclusive_scan_256(const float* first, const float* last, float* out)
    {
      static const __m256i him = _mm256_set1_epi32(7);
      const auto chunks = static_cast<int>(std::distance(first, last)) >> 3;
      __m256 sumv = _mm256_setzero_ps();
      for (int c = 0; c < chunks; ++c, first += 8, out += 8) {
        __m256 v = inclusive_scan_ps(_mm256_loadu_ps(first));
        sumv = _mm256_add_ps(sumv, v);
        _mm256_storeu_ps(out, sumv);
        sumv = _mm256_permutevar8x32_ps(sumv, him);
      }
      // scalar tail
      auto sum = chunks ? *(out - 1) : 0.0f;
      for (; first != last; ++first, ++out) {
        *out = (sum += *first);
      }
      return sum;
    }


    // returns sum
    inline float inclusive_scan_mul_256(const float* first, const float* last, const float* factor, float* out)
    {
      static const __m256i permctl = _mm256_set1_epi32(7);
      const auto chunks = static_cast<int>(std::distance(first, last)) >> 3;
      __m256 sumv = _mm256_setzero_ps();
      for (int c = 0; c < chunks; ++c, first += 8, factor += 8, out += 8) {
        const __m256 x = _mm256_mul_ps(_mm256_loadu_ps(first), _mm256_loadu_ps(factor));
        __m256 v = inclusive_scan_ps(x);
        sumv = _mm256_add_ps(sumv, v);
        _mm256_storeu_ps(out, sumv);
        sumv = _mm256_permutevar8x32_ps(sumv, permctl);
      }
      // scalar tail
      auto sum = chunks ? *(out - 1) : 0.0f;
      for (; first != last; ++first, ++factor, ++out) {
        *out = (sum += (*first * *factor));
      }
      return sum;
    }


    inline float inclusive_scan(const float* first, const float* last, float* out)
    {
      return inclusive_scan_256(first, last, out);
    }

    inline float inclusive_scan_mul(const float* first, const float* last, const float* factor, float* out)
    {
      return inclusive_scan_mul_256(first, last, factor, out);
    }

#else  // JC_SIMD

    template <typename IT, typename IT2, typename OIT>
    inline real_t inclusive_scan_mul(IT first, IT last, IT2 factor, OIT out)
    {
      auto sum = real_t(0);
      for (; first != last; ++first, ++factor, ++out) {
        *out = (sum += *factor * *first);
      }
      return sum;
    }

#endif  // JC_SIMD
  }
}

#endif
