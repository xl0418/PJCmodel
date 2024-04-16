#include <iostream>
#include <filesystem>
#include "game_watches.hpp"
#include "observer.hpp"
#include "jc.h"
#include "sampler.h"
#include "iomatlab.h"


namespace jc {

  class CommandLineObs : public Observer
  {
  public:
    CommandLineObs(const Parameter& param, std::unique_ptr<Observer> next) :
      Observer(std::move(next))
    {
      twatch_.start();
    }

  protected:
    void do_notify(ObserverMsg msg, int64_t T, const tick_stats* ts, const Model* model) override
    {
      switch (msg) {
      case ObserverMsg::Tick:
        ++ticks_;
        ts_ = ts_ + *ts;
        break;
      case ObserverMsg::ParsingContinuation:
        std::cout << "parsing continuation file\n";
        break;
      case ObserverMsg::Initialized:
        std::cout << "starting simulation: " << model->param().simId + 1;
        std::cout << "\nusing " << Sampler::get_name(model->param());
#ifdef JC_SIMD
        std::cout << " " << JC_SIMD;
#endif
        std::cout << '\n';
        watch_.start();
        break;
      default: 
        break;
      }
      if (watch_.elapsed<std::chrono::milliseconds>().count() > 2000) {
        watch_.stop();
        const auto& R = model->R();
        auto maxR = std::distance(R.cbegin(), std::max_element(R.cbegin(), R.cend()));
        auto junk = std::count_if(R.cbegin(), R.cend(), [](auto& a) { return a < 10; });
        std::cout << model->M().n() << ' ' << T << " +" << ticks_ << "\t " << R.size() << " +" << ts_.spec << " -" << ts_.ext << " \t";
        for (size_t i = 0; i < std::min(size_t(8), R.size()); ++i) {
          std::cout << R[i] << ' ';
        }
        std::cout << "  [" << maxR << ' ' << R[maxR] << "] [" << junk << "]  ";
        auto tt = watch_.elapsed<std::chrono::microseconds>().count() / ticks_;
        if (tt < 10000) {
          std::cout << tt << " us  ";
        }
        else {
          std::cout << tt / 1000 << " ms  ";
        }
        std::cout << watch_.elapsed<std::chrono::nanoseconds>().count() / ts_.G << " ns";
        std::cout << std::endl;
        ts_ = tick_stats{};
        ticks_ = 0;
        watch_.restart();
      }
      if (msg == ObserverMsg::Finished) {
        std::cout << "simulation " << model->param().simId + 1 << " finished in ";
        std::cout << twatch_.elapsed<std::chrono::seconds>().count() << " s" << std::endl;
      }
    }

  private:
    tick_stats ts_;
    size_t ticks_ = 0;
    game_watches::stop_watch<> twatch_;
    game_watches::stop_watch<> watch_;
  };


  class LogObs : public Observer
  {
  public:
    LogObs(const Parameter& param, std::unique_ptr<Observer> next) :
      Observer(std::move(next)),
      logger_(new MatLogger(param))
    {
    }

  protected:
    void do_notify(ObserverMsg msg, int64_t T, const tick_stats* ts, const Model* model) override
    {
      switch (msg) {
      case ObserverMsg::Tick:
        if ((T == model->param().totalTicks) || (!model->events().empty() && (model->param().log > 0) && (model->events().size() % model->param().log == 0))) {
          logger_->logSnapshot(T, model->gdm(), model->R());
        }
        break;
      case ObserverMsg::Finished:
        logger_->logLastSnapshot(T, model->M(), model->gdm(), model->R());
        logger_->logEvents(model->events());
        logger_->logState(model->param());
        break;
      default:
        break;
      }
    }

  private:
    std::unique_ptr<MatLogger> logger_;
  };


  class ProfileObs : public Observer
  {
  public:
    ProfileObs(const Parameter& param, std::unique_ptr<Observer> next) :
      Observer(std::move(next))
    {}

  protected:
    void do_notify(ObserverMsg msg, int64_t T, const tick_stats* ts, const Model* model) override
    {
      switch (msg) {
      case ObserverMsg::Tick:
        if (profile_.empty() || (profile_.back().R != model->R().size())) {
          auto dt = watch_.elapsed<std::chrono::nanoseconds>().count();
          profile_.push_back(entry{
            model->M().n(),
            model->R().size(),
            ts->G,
            model->param().jc_cutoff,
            model->param().disp_cutoff,
            dt,
            dt / static_cast<long long>(ts->G)
          });
        }
        watch_.restart();
        break;
      case ObserverMsg::Initialized:
        watch_.start();
        break;
      case ObserverMsg::Finished: {
        auto os = (model->param().filename.empty()) 
          ? std::ofstream("profile.profile")
          : std::ofstream(std::filesystem::path(model->param().filename).replace_extension("profile"));
        for (size_t i = 1; i < profile_.size(); ++i) {
          const auto& e = profile_[i];
          os << e.L << ' ' << e.R << ' ' << e.G << ' ' << e.jc_cutoff << ' '
             << e.disp_cutoff << ' ' << e.duration << ' ' << e.t_duration << '\n';
        }
        break;
      }
      default:
        break;
      }
    }

  private:
    struct entry {
      size_t L;
      size_t R;
      size_t G;
      int jc_cutoff;
      int disp_cutoff;
      long long duration;
      long long t_duration;
    };
    std::vector<entry> profile_;
    game_watches::stop_watch<> watch_;
  };


  class CheckObs : public Observer
  {
  public:
    CheckObs(const Parameter& param, std::unique_ptr<Observer> next) :
      Observer(std::move(next)),
      interval_(param.checkInterval)
    {
    }

  protected:
    void do_notify(ObserverMsg msg, int64_t T, const tick_stats* ts, const Model* model) override
    {
      switch (msg) {
      case ObserverMsg::Tick:
        if (watch_.elapsed<std::chrono::seconds>() > interval_) {
          R_.assign(model->R().size(), 0);
          for_each(make_block_view(model->M()), [&](int sp) {
            ++R_[sp];
          });
          if (R_ != model->R()) {
            throw std::runtime_error("R-M mismatch");
          }
          watch_.restart();
        }
        break;
      case ObserverMsg::Initialized:
        watch_.start();
        break;
      default:
        break;
      }
    }

  private:
    game_watches::stop_watch<> watch_;
    std::chrono::seconds interval_;
    std::vector<int> R_;
  };


  class SummaryObs : public Observer
  {
  public:
    SummaryObs(const Parameter& param, std::unique_ptr<Observer> next) :
      Observer(std::move(next))
    {
    }

  protected:
    void do_notify(ObserverMsg msg, int64_t T, const tick_stats* ts, const Model* model) override
    {
      switch (msg) {
      case ObserverMsg::Tick:
        ts_.ext += ts->ext;
        ts_.spec += ts->spec;
        if (0 == T % 1000) {
          summary_.push_back({ T, ts_ });
          ts_.spec = ts_.ext = 0;
        }
        break;
      case ObserverMsg::Finished: {
        auto os = (model->param().filename.empty())
          ? std::ofstream("summary.summary")
          : std::ofstream(std::filesystem::path(model->param().filename).replace_extension("summary"));
        for (const auto& s : summary_) {
          os << s.T << ' ' << s.ts.spec << ' ' << s.ts.ext << '\n';
        }
        break;
      }
      default:
        break;
      }
    }

  private:
    struct summary_t { int64_t T; tick_stats ts; };
    tick_stats ts_ = { 0,0,0 };
    std::vector<summary_t> summary_;
  };


  std::unique_ptr<Observer> create_cmd_line_obs(const Parameter& param, std::unique_ptr<Observer>&& next)
  {
    return std::make_unique<CommandLineObs>(param, std::move(next));
  }

  std::unique_ptr<Observer> create_log_obs(const Parameter& param, std::unique_ptr<Observer>&& next)
  {
    return std::make_unique<LogObs>(param, std::move(next));
  }

  std::unique_ptr<Observer> create_profile_obs(const Parameter& param, std::unique_ptr<Observer>&& next)
  {
    return std::make_unique<ProfileObs>(param, std::move(next));
  }

  std::unique_ptr<Observer> create_check_obs(const Parameter& param, std::unique_ptr<Observer>&& next)
  {
    return std::make_unique<CheckObs>(param, std::move(next));
  }

  std::unique_ptr<Observer> create_summary_obs(const Parameter& param, std::unique_ptr<Observer>&& next)
  {
    return std::make_unique<SummaryObs>(param, std::move(next));
  }

}
