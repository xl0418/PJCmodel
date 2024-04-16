/*! \file square_buffer.h
*  \brief 2D memory buffer
*  \author Hanno Hildenbrandt
*/

#ifndef JC_SQUARE_BUFFER_H_INCLUDED
#define JC_SQUARE_BUFFER_H_INCLUDED

#include <cmath>
#include <cassert>
#include <cstddef>
#include <memory>
#include <array>
#include <utility>
#include <random>
#include <type_traits>
#include <iterator>
#include <tbb/tbb.h>
#include "jc_config.hpp"
#ifdef JC_CPP20
# include <bit>
#endif


namespace jc {


  struct position
  {
    int x, y;
  };


  template <typename RENG>
  inline position randomPos(int L, RENG& reng)
  {
    auto pdist = std::uniform_int_distribution<int>(0, L - 1);
    return{ pdist(reng), pdist(reng) };
  }


  inline bool operator<(const position& a, const position& b)
  {
#ifdef JC_CPP20
    return std::bit_cast<uint64_t>(a) < std::bit_cast<uint64_t>(b);
#else
    const auto la = (ptrdiff_t(a.y) << 32) | a.x;
    const auto lb = (ptrdiff_t(b.y) << 32) | b.x;
    return la < lb;
#endif
  }

  inline bool operator==(const position& a, const position& b)
  {
#ifdef JC_CPP20
    return std::bit_cast<uint64_t>(a) == std::bit_cast<uint64_t>(b);
#else
    return (a.x == b.x) && (a.y == b.y);
#endif
  }

  inline bool operator!=(const position& a, const position& b)
  {
    return !(a == b);
  }


  struct torus
  {
    explicit torus(int L) : L_(L) {}

    constexpr int wrap(int c) const noexcept
    {
      return (c < 0) ? L_ + c : (c >= L_) ? c - L_ : c;
    }

    constexpr position wrap(const position& pos) const noexcept
    {
      return { wrap(pos.x), wrap(pos.y) };
    }

    // shortest 1-dim distance square on torus
    /* constexpr */ int dist2(int c0, int c1) const noexcept
    {
      const auto ad = std::abs(c0 - c1);
      const auto d = std::min(ad, L_ - ad);
      return d * d;
    }

    // returns pair of inclusive ranges [begin, end)
    // first range is the bigger one
    struct range { int begin = 0; int end = 0; };
    using ranges = std::pair<range, range>;
    ranges kernel_ranges(int center, int r) const noexcept
    {
      assert(r < (L_ >> 1));
      assert(center == wrap(center));
      using std::get;
      auto ret = ranges{};
      if ((wrap(center - r) == (center - r)) && (wrap(center + r) == (center + r))) {
        get<0>(ret) = { center - r, center + r + 1 };
      }
      else if (wrap(center - r) == (center - r)) {
        get<0>(ret) = { 0, wrap(center + r) + 1 };
        get<1>(ret) = { center - r, L_ };
      }
      else {
        get<0>(ret) = { 0, center + r + 1 };
        get<1>(ret) = { wrap(center - r), L_ };
      }
      // bring bigger range to front
      if ((get<0>(ret).end - get<0>(ret).begin) < (get<1>(ret).end - get<1>(ret).begin)) {
        std::swap(get<0>(ret), get<1>(ret));
      }
      return ret;
    }

    const int L_;
  };


  template <typename T>
  class square_buffer
  {
  public:
    square_buffer() {}

    square_buffer(const square_buffer& rhs) : buf_(new T[rhs.s_ * rhs.s_]), n_(rhs.n_), s_(rhs.s_)
    {
      copy(make_block_view(*this), make_block_view(rhs));
    }

    square_buffer& operator=(const square_buffer& rhs)
    {
      auto tmp = square_buffer(rhs);
      swap(tmp);
      return *this;
    }

    explicit square_buffer(size_t n) : buf_(new T[n * n]), n_(n), s_(n) {}

    square_buffer(size_t n, T val) : square_buffer(n)
    {
      std::fill_n(data(), n * n, val);
    }

    square_buffer(size_t n, size_t stride, bool) :
      square_buffer(stride)
    {
      resize(n);
    }

    square_buffer(size_t n, size_t stride, T val, bool) :
      square_buffer(stride, val)
    {
      resize(n);
    }

    // destructive if reallocation is neccessary
    void resize(size_t n)
    {
      if (s_ < n) {
        auto nn = n + 64;
        buf_.reset(nullptr);
        buf_.reset(new T[nn * nn]);
        s_ = nn;
      }
      n_ = n;
    }

    T operator()(size_t i, size_t j) const noexcept { return buf_[i + j * s_]; }
    T& operator()(size_t i, size_t j) noexcept { return buf_[i + j * s_]; }
    T operator()(const position& pos) const noexcept { return buf_[pos.x + pos.y * s_]; }
    T& operator()(const position& pos) noexcept { return buf_[pos.x + pos.y * s_]; }

    bool empty() const noexcept { return n_ == 0; }
    size_t n() const noexcept { return n_; }
    size_t stride() const noexcept { return s_; }
    size_t elements() const noexcept { return n_ * n_; }
    auto torus() const noexcept { return jc::torus{ static_cast<int>(n_) }; }

    T* data() noexcept { return buf_.get(); }
    const T* data() const noexcept { return buf_.get(); }

    T* begin() noexcept { assert(n_ == s_); return data(); }
    const T* cbegin() const noexcept { assert(n_ == s_); return data(); }
    T* end() noexcept { assert(n_ == s_); return data() + elements(); }
    const T* cend() const noexcept { assert(n_ == s_); return data() + elements(); }

    T operator[](size_t i) const { return cbegin()[i]; }
    T& operator[](size_t i) { return begin()[i]; }

    T* row(size_t i) noexcept { return data() + i * s_; }
    const T* row(size_t i) const noexcept { return data() + i * s_; }
    
    void swap(square_buffer& rhs)
    {
      using std::swap;
      swap(buf_, rhs.buf_);
      swap(n_, rhs.n_);
      swap(s_, rhs.s_);
    }

  private:
    std::unique_ptr<T[]> buf_;
    size_t n_ = 0;
    size_t s_ = 0;
  };


  template <typename IT>
  inline auto make_square_buffer(IT first, size_t n)
  {
    auto ret = square_buffer<typename std::iterator_traits<IT>::value_type>(n);
    std::copy_n(first, n * n, ret.data());
    return ret;
  }

    
  template <typename T>
  class block_view
  {
  public:
    block_view(T* data, size_t w, size_t h, size_t s) : data_(data), w_(w), h_(h), s_(s)
    {}

    size_t width() const noexcept { return w_; }
    size_t height() const noexcept { return h_; }
    size_t stride() const noexcept { return s_; }

    std::remove_cv_t<T>* operator()(size_t i, size_t j) const noexcept { return data_[i + j * s_]; }
    T& operator()(size_t i, size_t j) noexcept { return data_[i + j * s_]; }
    std::remove_cv_t<T>* operator()(const position& pos) const noexcept { return data_[pos.x + pos.y * s_]; }
    T& operator()(const position& pos) noexcept { return data_[pos.x + pos.y * s_]; }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    auto shift(ptrdiff_t x, ptrdiff_t y) const noexcept
    {
      return block_view(data_ + s_ * y + x, w_, h_, s_);
    }

  private:
    T* data_ = nullptr;
    size_t w_ = 0;
    size_t h_ = 0;
    size_t s_ = 0;
  };

  template <typename T>
  inline auto make_block_view(T* data, size_t w, size_t h, size_t s)
  {
    return block_view<T>(data, w, h, s);
  }


  template <typename T>
  inline auto make_block_view(square_buffer<T>& sb, size_t w, size_t h)
  {
    return block_view<T>(sb.data(), w, h, sb.stride());
  }


  template <typename T>
  inline auto make_block_view(const square_buffer<T>& sb, size_t w, size_t h)
  {
    return block_view<std::add_const_t<T>>(sb.data(), w, h, sb.stride());
  }

  template <typename T>
  inline auto make_block_view(square_buffer<T>& sb)
  {
    return block_view<T>(sb.data(), sb.n(), sb.n(), sb.stride());
  }


  template <typename T>
  inline auto make_block_view(const square_buffer<T>& sb)
  {
    return block_view<std::add_const_t<T>>(sb.data(), sb.n(), sb.n(), sb.stride());
  }


  // 'copy to the left' is safe for overlapping blocks
  template <typename T, typename T1>
  inline void copy(block_view<T> dst, block_view<T1> src)
  {
    static_assert(std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<T1>>, "");
    assert((dst.height() == src.height()) && (dst.width() == src.width()));
    auto pdst = dst.data();
    auto psrc = src.data();
    for (size_t j = 0; j < src.height(); ++j, pdst += dst.stride(), psrc += src.stride()) {
      for (size_t i = 0; i < src.width(); ++i) {
        pdst[i] = psrc[i];
      }
    }
  }


  // dst and src shall not overlap
  template <typename T, typename T1>
  inline void parallel_copy(block_view<T> dst, block_view<T1> src)
  {
    static_assert(std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<T1>>, "");
    assert((dst.height() == src.height()) && (dst.width() == src.width()));
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, src.height(), 0, src.width()), [&](const auto& r) {
      auto pdst = dst.data() + r.rows().begin() * dst.stride();
      auto psrc = src.data() + r.rows().begin() * src.stride();
      for (size_t j = r.rows().begin(); j != r.rows().end(); ++j, pdst += dst.stride(), psrc += src.stride()) {
        for (size_t i = r.cols().begin(); i != r.cols().end(); ++i) {
          pdst[i] = psrc[i];
        }
      }
    });
  }


  template <typename T, typename Fun>
  inline void for_each(block_view<T> bv, Fun&& fun)
  {
    auto p = bv.data();
    for (size_t j = 0; j < bv.height(); ++j, p += bv.stride()) {
      for (size_t i = 0; i < bv.width(); ++i) {
        fun(p[i]);
      }
    }
  }


  template <typename T, typename Fun>
  inline void for_each_xy(block_view<T> bv, Fun&& fun)
  {
    auto p = bv.data();
    for (int j = 0; j < static_cast<int>(bv.height()); ++j, p += bv.stride()) {
      for (int i = 0; i < static_cast<int>(bv.width()); ++i) {
        fun(p[i], i, j);
      }
    }
  }


  template <typename T, typename Fun>
  inline void for_each_xy(const square_buffer<T>& buf, Fun&& fun)
  {
    auto p = buf.data();
    const auto k = static_cast<int>(buf.n());
    for (int j = 0; j < k; ++j, p += buf.stride()) {
      for (int i = 0; i < k; ++i) {
        fun(p[i], i, j);
      }
    }
  }


  template <typename T, typename Fun>
  inline void parallel_for_each(block_view<T> bv, Fun&& fun)
  {
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, bv.height(), 0, bv.width()), [&](auto r) {
      auto pr = bv.data() + r.rows().begin() * bv.stride();
      for (size_t j = r.rows().begin(); j != r.rows().end(); ++j, pr += bv.stride()) {
        for (size_t i = r.cols().begin(); i != r.cols().end(); ++i) {
          fun(pr[i]);
        }
      }
    });
  }


  template <typename T, typename Fun>
  inline void parallel_for_each_xy(block_view<T> bv, Fun&& fun)
  {
    tbb::parallel_for(tbb::blocked_range2d<size_t>(0, bv.height(), 0, bv.width()), [&](auto r) {
      auto pr = bv.data() + r.rows().begin() * bv.stride();
      for (size_t j = r.rows().begin(); j != r.rows().end(); ++j, pr += bv.stride()) {
        for (size_t i = r.cols().begin(); i != r.cols().end(); ++i) {
          fun(pr[i], static_cast<int>(i), static_cast<int>(j));
        }
      }
    });
  }


} // namespace jc

#endif
