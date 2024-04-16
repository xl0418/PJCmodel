#ifndef JC_OBSERVER_HPP_INCLUDED
#define JC_OBSERVER_HPP_INCLUDED

#include <memory>


namespace jc {


  enum class ObserverMsg {
    Tick = 0,
    Initialized,
    Finished,
    ParsingContinuation
  };

  // ext, spec, G
  struct tick_stats 
  {
    size_t ext = 0;
    size_t spec = 0;
    size_t G = 0;
  };
  
  
  inline tick_stats operator+(const tick_stats& a, const tick_stats& b)
  {
    return { a.ext + b.ext, a.spec + b.spec, a.G + b.G };
  }


  class Observer
  {
  public:
    Observer() {}
    explicit Observer(std::unique_ptr<Observer>&& next) : next_(std::move(next)) {}
    virtual ~Observer() {}

    void notify(ObserverMsg msg, int64_t T, const tick_stats* ts, const class Model* model)
    {
      do_notify(msg, T, ts, model);
      if (next_) next_->notify(msg, T, ts, model);
    }
  
  protected:
    virtual void do_notify(ObserverMsg msg, int64_t T, const tick_stats* ts, const class Model*) {};

  private:
    std::unique_ptr<Observer> next_;
  };


  std::unique_ptr<Observer> create_cmd_line_obs(const struct Parameter& param, std::unique_ptr<Observer>&& next);
  std::unique_ptr<Observer> create_log_obs(const struct Parameter& param, std::unique_ptr<Observer>&& next);
  std::unique_ptr<Observer> create_profile_obs(const struct Parameter& param, std::unique_ptr<Observer>&& next);
  std::unique_ptr<Observer> create_check_obs(const Parameter& param, std::unique_ptr<Observer>&& next);
  std::unique_ptr<Observer> create_summary_obs(const Parameter& param, std::unique_ptr<Observer>&& next);

}

#endif
