#ifndef JC_JC_CONFIG_HPP_INCLUDED
#define JC_JC_CONFIG_HPP_INCLUDED


//#define JC_NO_AVX512      // avoid AVX512
//#define JC_NO_AVX2        // avoid AVX2
//#define JC_SCAN_512


#if defined(__AVX512F__) && !defined(JC_NO_AVX512)
# define JC_AVX2 1
# define JC_AVX512 1
# define JC_SIMD "AVX512_256"
#elif defined(__AVX2__) && !defined(JC_NO_AVX2)
# define JC_AVX2 1
# define JC_SIMD "AVX2"
# undef JC_SCAN_512
#else
# undef JC_SIMD
# undef JC_SCAN_512
#endif

#if ((defined(_MSVC_LANG) && _MSVC_LANG == 201703L) || __cplusplus == 201703L)
# define JC_CPP17 1
#endif
#if ((defined(_MSVC_LANG) && _MSVC_LANG > 201703L) || __cplusplus > 201703L)
# define JC_CPP20 1
#endif


#include <type_traits>


namespace jc {

  static_assert(std::is_same_v<int, int32_t>, "int shall be 32bit");
  constexpr char Version[] = "jc 2.2.0";

  using real_t = float;

  // maximum permitted events per timestep
  constexpr int64_t max_sequence_number = 1000;

}


#ifdef JC_SIMD
#include "jc_simd.h"
#endif


#endif
