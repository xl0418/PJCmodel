#include <iostream>
#include <numeric>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <tbb/tbb.h>
#include "jc.h"
#include "sampler.h"
#include "iomatlab.h"
#include "observer.hpp"
#include "game_watches.hpp"


namespace jc {

  namespace {

    // thread local storage

    // the simulation id.
    std::atomic<int> tls_simId = -1;

    struct model_tls_t
    {
      explicit model_tls_t(const Parameter& param) :
        sampler(Sampler::create(param)),
        reng(rndutils::make_random_engine<>()),
        simId(param.simId)
      {}

      std::unique_ptr<Sampler> sampler;
      rndutils::default_engine reng;
      int simId;
    };
    thread_local std::unique_ptr<model_tls_t> model_tls;


    model_tls_t& get_tls(const Parameter& param)
    {
      auto& tls = model_tls;
      // we need to create a new thread local storage 
      // - if this is the first attempt of the calling thread.
      // - if the calling thread runs a different simId.
      if (!tls || (tls->simId != tls_simId.load(std::memory_order_acquire))) {
        tls = std::make_unique<model_tls_t>(param);   // phoenix
      }
      return *tls;
    }
  
  } // anonymous namespace

  
  Kernel make_spar_kernel(const Parameter& param)
  {
    if (param.s_jc <= 0.0) return Kernel(square_buffer<real_t>{});
    const auto r = param.jc_cutoff;
    const auto k = 2 * r + 1;
    square_buffer<real_t> K(k, 0);

    const auto ss = std::max(1e-20, 2.0 * param.s_jc * param.s_jc);
    for (int y = 0; y < k; ++y) {
      const auto yy = (r - y) * (r - y);
      for (int x = 0; x < k; ++x) {
        const auto dd = (r - x) * (r - x) + yy;
        K(x, y) = (dd && (dd <= r * r)) ? static_cast<real_t>(std::exp(-double(dd) / ss)) : real_t(0);     // mask center
      }
    }
    double sum = 0.0;
    for (auto x : K) {
      sum += static_cast<double>(x);
    }
    const auto scale = static_cast<real_t>(param.Psi / sum);
    for (auto& k : K) {
      k *= scale;
    }
    return Kernel(std::move(K));
  }


  Kernel make_dispersal_kernel(const Parameter& param)
  {
    if (param.s_disp <= 0.0) return Kernel(square_buffer<real_t>{});
    const auto r = param.disp_cutoff;
    const auto k = 2 * r + 1;
    square_buffer<real_t> K(k, 0);

    const auto ss = std::max(1e-20, 2.0 * param.s_disp * param.s_disp);
    for (int y = 0; y < k; ++y) {
      const auto yy = (r - y) * (r - y);
      for (int x = 0; x < k; ++x) {
        const auto dd = (r - x) * (r - x) + yy;
        K(x, y) = (dd && (dd <= r * r)) ? static_cast<real_t>(std::exp(-double(dd) / ss)) : real_t(0);     // mask center
      }
    }
    return Kernel(std::move(K));
  }
  

  // class Model is required to set tls_simId at construction.
  // this helper guaranties this
  Model::tls_guard::tls_guard(const Parameter& param) noexcept
  {
    tls_simId.store(param.simId, std::memory_order_release);
  }


  Model::Model(const Parameter& param) :
    param_(param),
    reng_(rndutils::make_random_engine<>()),
    T0_(0),
    M_(size_t(param.L), 0),
    R_({ param.L0 * param.L0 - 1, 1 }),
    kernels_(param),
    gdm_(std::make_unique<GDM>(param)),
    tls_guard_(param)
  {
    M_.resize(param.L0);
    param_.totalTicks = param_.ticks;
    if (param_.cont.empty()) {
      // place singleton of species 1
      M_(0, 0) = 1;
    }
    else {
      Continuation cont(param.cont);
      if (cont.param_.max_seq_num != jc::max_sequence_number) throw std::runtime_error("continuation: max events per time step mismatch");
      if (param_.L != cont.param_.L) throw std::runtime_error("continuation: L mismatch");
      if (cont.M_.size() != M_.elements()) throw std::runtime_error("continuation file corrupted");
      std::copy(cont.M_.cbegin(), cont.M_.cend(), M_.begin());
      R_ = cont.R_;
      if (R_.size() * R_.size() != cont.D_.size()) throw std::runtime_error("continuation file corrupted");
      convert_R();
      gdm_ = std::make_unique<GDM>(param_, Rf_, make_square_buffer(cont.D_.cbegin(), R_.size()));
      T0_ = cont.param_.totalTicks;
      param_.totalTicks += cont.param_.totalTicks;
    }
  }


  void Model::generate_random_positions()
  {
    const auto L = static_cast<int>(M_.n());
    const auto LL = double(M_.elements());
    auto G = static_cast<size_t>(param_.mu * LL); // std::poisson_distribution<int>(param_.mu * LL)(reng_);
    sampler_pos_.clear();
    sampler_events_.clear();
    while (sampler_pos_.size() < G) {
      auto it = sampler_pos_.insert(randomPos(L, reng_));
      if (it.second) {
        sampler_events_.push_back( { *it.first } );
      }
    }
  }


  void Model::shift_random_positions()
  {
    auto G = static_cast<int>(sampler_events_.size());
    const auto L = static_cast<int>(M_.n());
    auto shift_dist = std::normal_distribution<>(0.0, param_.mu * L);
    const position shift = { 
      std::clamp(static_cast<int>(shift_dist(reng_)), -L / 4, +L / 4),
      std::clamp(static_cast<int>(shift_dist(reng_)), -L / 4, +L / 4)
    };
    const auto torus = jc::torus(L);
    const auto transpose = false;// std::bernoulli_distribution(0.5)(reng_);
    if (transpose) {
      tbb::parallel_for(tbb::blocked_range<int>(0, G), [&](auto& r) {
        for (auto i = r.begin(); i < r.end(); ++i) {
          const auto& p = sampler_events_[i].pos;
          const position ps = { torus.wrap(p.x + shift.x), torus.wrap(p.y + shift.y) };
          sampler_events_[i] = { ps, M_(ps.y, ps.x), -1 };
        }
      });
    }
    else {
      tbb::parallel_for(tbb::blocked_range<int>(0, G), [&](auto& r) {
        for (auto i = r.begin(); i < r.end(); ++i) {
          const auto& p = sampler_events_[i].pos;
          const position ps = { torus.wrap(p.y + shift.y), torus.wrap(p.x + shift.x) };
          sampler_events_[i] = { ps, M_(ps.x, ps.y), -1 };
        }
      });
    }
  }


  void Model::convert_R()
  {
    Rf_.resize(R_.size());
    for (size_t i = 0; i < R_.size(); ++i) {
      Rf_[i] = real_t(R_[i]);
    }
  }


  tick_stats Model::timestep(int64_t T)
  {
    size_t extinctions = 0;
    size_t speciations = 0;
    (0 == T % 10000) 
      ? generate_random_positions() 
      : shift_random_positions();

    // sample intruders
    auto G = static_cast<int>(sampler_events_.size());
    tbb::parallel_for(0, G, [&](int g) {
      auto& tls = get_tls(param_);
      sampler_events_[g].rsp = tls.sampler->draw(sampler_events_[g].pos, { M_, *gdm_, Rf_ }, kernels_, tls.reng);
    });

    // takeover of vacant spots
    for (auto g = 0; g < G; ++g) {
      auto [pos, sp, rsp] = sampler_events_[g];
      --R_[sp];
      ++R_[rsp];
      M_(pos.x, pos.y) = rsp;
    }

    int64_t seq = 0;    // sequence number
    const auto Nu = std::poisson_distribution<int>(param_.nu * G)(reng_);
    for (auto nu = 0; nu < Nu; ++nu) {
      // we have an specification event
      ++speciations;
      const auto pos = randomPos(static_cast<int>(M_.n()), reng_);
      const auto sp = M_(pos.x, pos.y);
      const auto nsp = static_cast<int>(R_.size());
      const auto n = gdm_->specification(pos, nsp, M_);
      R_.push_back(n);
      R_[sp] -= n;
      events_.push_back(event{ param_.max_seq_num * T + seq++, R_.size(), pos, nsp, sp });
    }

    int i = 0;
    while (i < static_cast<int>(R_.size())) {
      if (R_[i] > 0) {
        ++i;
      }
      else {
        // we have an extinction event
        ++extinctions;
        gdm_->extinction(i);                             // Remove species sp from GDM
        R_.erase(R_.begin() + i);                // Remove abundance entry for species sp
        parallel_for_each(make_block_view(M_), [sp = i](auto& x) { if (x > sp) --x; });  // Adjust species id
        events_.push_back(event{ param_.max_seq_num * T + seq++, R_.size(), {0,0}, i, -1 });
      }
    }
    if (seq > jc::max_sequence_number) {
      throw std::runtime_error("too many events per time step.");
    }
    return { extinctions, speciations, size_t(G) };
  }


  void Model::area_growth(int64_t T)
  {
    for (int inc = 0; inc < param_.Linc; ++inc) {
      const auto L = static_cast<int>(M_.n());
      const auto G = 2 * L + 1;

      area_growth_events_.resize(G);
      for (int i = 0; i < L; ++i) {
        area_growth_events_[i].pos = { i, L - 1 };
        area_growth_events_[i + L].pos = { L - 1, i };
      }
      area_growth_events_.back() = { { L - 1, L - 1 } };

      area_growth_events_.resize(G);
      tbb::parallel_for(0, G, [&](int g) {
        auto& tls = get_tls(param_);
        area_growth_events_[g].rsp = tls.sampler->draw(area_growth_events_[g].pos, { M_, *gdm_, Rf_ }, kernels_, tls.reng);
      });

      // takover of vaccant spots
      M_.resize(L + 1);
      for (int i = 0; i < L; ++i) {
        const auto rsp0 = M_(i, L) = area_growth_events_[i].rsp;
        const auto rsp1 = M_(L, i) = area_growth_events_[i + L].rsp;
        ++R_[rsp0];
        ++R_[rsp1];
      }
      const auto rsp = M_(L, L) = area_growth_events_.back().rsp;
      ++R_[rsp];
      if (L == param_.L) {
        break;
      }
    }
  }


  int64_t Model::run(Observer* obs)
  {
    obs->notify(ObserverMsg::Initialized, 0, nullptr, this);
    generate_random_positions();
    for (auto T = T0_; T < param_.totalTicks; ++T) {
      convert_R();
      gdm_->update(int(M_.n()), Rf_, 1);
      if ((int(M_.n()) != param_.L) && (0 == T % param_.Lg)) {
        area_growth(T);
        generate_random_positions();
      }
      auto ts = timestep(T);
      obs->notify(ObserverMsg::Tick, T, &ts, this);
    }
    obs->notify(ObserverMsg::Finished, param_.totalTicks, nullptr, this);
    return param_.ticks;
  }


  kernels_t::kernels_t(const Parameter& param) :
    Kdisp(make_dispersal_kernel(param)),
    Kjc(make_spar_kernel(param))
  {
  }


} // namespace jc
