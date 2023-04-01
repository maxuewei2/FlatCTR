#include <chrono>
#include <fstream>
#include <iomanip>
#include <pthread.h>
#include <sstream>

#include "cxxopts.hpp"
#include "spdlog/spdlog.h"

#include "blocking_queue/blocking_queue.h"
#include "common.h"
#include "dataset/parser.h"
#include "metric.h"
#include "model/lr_model.h"
#include "model/fm_model.h"

using namespace std;

typedef chrono::high_resolution_clock            Time;
typedef chrono::time_point<chrono::system_clock> Clock;

struct Config
{
  string   model;
  string   train_file;
  string   valid_file;
  string   test_file;
  string   test_pred_file;
  string   load;
  string   save;
  F        w_lr;
  F        v_lr;
  F        w_l2;
  F        v_l2;
  F        v_stddev;
  uint32_t epoch;
  uint32_t batch_size;
  uint32_t k;
  uint32_t train_thread_num;
  uint32_t parse_thread_num;
  long     seed;
  bool     debug;

  [[nodiscard]] string str() const
  {
    char ss[10000] = "";
    int  padding   = 20;
    sprintf(ss + strlen(ss), "%*s: %s\n", padding, "model", model.c_str());
    sprintf(ss + strlen(ss), "%*s: %s\n", padding, "train_file", train_file.c_str());
    sprintf(ss + strlen(ss), "%*s: %s\n", padding, "valid_file", valid_file.c_str());
    sprintf(ss + strlen(ss), "%*s: %s\n", padding, "test_file", test_file.c_str());
    sprintf(ss + strlen(ss), "%*s: %s\n", padding, "test_pred_file", test_pred_file.c_str());
    sprintf(ss + strlen(ss), "%*s: %s\n", padding, "load", load.c_str());
    sprintf(ss + strlen(ss), "%*s: %s\n", padding, "save", save.c_str());
    sprintf(ss + strlen(ss), "%*s: %.9g\n", padding, "w_lr", w_lr);
    sprintf(ss + strlen(ss), "%*s: %.9g\n", padding, "v_lr", v_lr);
    sprintf(ss + strlen(ss), "%*s: %.9g\n", padding, "w_l2", w_l2);
    sprintf(ss + strlen(ss), "%*s: %.9g\n", padding, "v_l2", v_l2);
    sprintf(ss + strlen(ss), "%*s: %.9g\n", padding, "v_stddev", v_stddev);
    sprintf(ss + strlen(ss), "%*s: %u\n", padding, "epoch", epoch);
    sprintf(ss + strlen(ss), "%*s: %u\n", padding, "batch_size", batch_size);
    sprintf(ss + strlen(ss), "%*s: %u\n", padding, "k", k);
    sprintf(ss + strlen(ss), "%*s: %u\n", padding, "train_thread_num", train_thread_num);
    sprintf(ss + strlen(ss), "%*s: %u\n", padding, "parse_thread_num", parse_thread_num);
    sprintf(ss + strlen(ss), "%*s: %ld\n", padding, "seed", seed);
    sprintf(ss + strlen(ss), "%*s: %d\n", padding, "debug", debug);

    return ss;
  }
} cfg;

void parse_thread(const int id, BlockingQueue<unique_ptr<vector<unique_ptr<string>>>>& line_queue,
                  BlockingQueue<unique_ptr<vector<unique_ptr<Sample>>>>& sample_queue)
{
  while (true)
  {
    unique_ptr<vector<unique_ptr<string>>> lines;
    line_queue.pop(lines);
    if (lines == nullptr) [[unlikely]]
    {
      break;
    }
    unique_ptr<vector<unique_ptr<Sample>>> samples = make_unique<vector<unique_ptr<Sample>>>();
    samples->reserve(cfg.batch_size);
    for (auto& line : *lines)
    {
      unique_ptr<Sample> sample = Parser::parseSample(std::move(line));
      if (cfg.debug) [[unlikely]]
        spdlog::debug("{}: SAMPLE\t {}", id, sample->to_string());
      samples->push_back(std::move(sample));
    }
    sample_queue.push(std::move(samples));
  }
  if (cfg.debug)
    spdlog::debug("parse thread {:4d} end", id);
}

void train_thread(const int id, Base* model,
                  BlockingQueue<unique_ptr<vector<unique_ptr<Sample>>>>& sample_queue)
{
  while (true)
  {
    unique_ptr<vector<unique_ptr<Sample>>> samples;
    sample_queue.pop(samples);
    if (samples == nullptr) [[unlikely]]
    {
      break;
    }
    model->learn(*samples);
  }
  if (cfg.debug)
    spdlog::debug("train thread {:4d} end", id);
}

int run()
{
  if (cfg.seed != -1)
    srand(cfg.seed);

  Clock                   t_begin, t_end;
  chrono::duration<float> cost{};

  Base* model;
  if (cfg.model == "lr")
    model = new LR(cfg.w_lr, cfg.w_l2);
  if (cfg.model == "fm")
    model = new FM(cfg.k, cfg.w_lr, cfg.v_lr, cfg.w_l2, cfg.v_l2, cfg.v_stddev, cfg.seed);

  /*********************************************************
  *  model loading                                         *
  *********************************************************/
  if (!cfg.load.empty())
  {
    t_begin = Time::now();
    spdlog::info("**************** load model ****************");
    spdlog::info("load from {}", cfg.load);
    size_t model_size = model->load(cfg.load);
    if (model_size == 0)
    {
      spdlog::error("error loading model {}", cfg.load);
      exit(-1);
    }
    t_end = Time::now();
    cost  = t_end - t_begin;
    spdlog::info("finish, num_feat: {}, costs {:.4f} secs", model_size, cost.count());
  }

  /*********************************************************
  *  training                                              *
  *********************************************************/
  {
    Parser parser_train(cfg.train_file);
    for (size_t epoch_i = 0; epoch_i < cfg.epoch; epoch_i++)
    {
      t_begin = Time::now();
      spdlog::info("******************************************************");
      int n_sample = 0, step = 1000000;

      BlockingQueue<unique_ptr<vector<unique_ptr<string>>>> line_queue(100);
      BlockingQueue<unique_ptr<vector<unique_ptr<Sample>>>> sample_queue(100);

      vector<thread> parse_threads;
      for (size_t i = 0; i < cfg.parse_thread_num; ++i)
      {
        parse_threads.emplace_back(parse_thread, i, ref(line_queue), ref(sample_queue));
        stringstream ss;
        ss << std::setfill('0') << std::setw(3) << "parse_" << i;
        pthread_setname_np(parse_threads[i].native_handle(), ss.str().c_str());
      }

      vector<thread> train_threads;
      for (size_t i = 0; i < cfg.train_thread_num; ++i)
      {
        train_threads.emplace_back(train_thread, i, model, ref(sample_queue));
        stringstream ss;
        ss << std::setfill('0') << std::setw(3) << "train_" << i;
        pthread_setname_np(train_threads[i].native_handle(), ss.str().c_str());
      }

      Clock last = Time::now();
      parser_train.reset();
      unique_ptr<vector<unique_ptr<string>>> lines = make_unique<vector<unique_ptr<string>>>();
      lines->reserve(cfg.batch_size);
      while (unique_ptr<string> line = parser_train.nextLine())
      {
        lines->push_back(std::move(line));
        if (lines->size() == cfg.batch_size)
        {
          line_queue.push(std::move(lines));
          lines = make_unique<vector<unique_ptr<string>>>();
          lines->reserve(cfg.batch_size);
        }
        n_sample++;
        if (n_sample % step == 0) [[unlikely]]
        {
          cost = Time::now() - last;
          spdlog::info("epoch {:4d}: {:8d} samples, {:.4f} secs", epoch_i, n_sample, cost.count());
          last = Time::now();
        }
      }
      if (!lines->empty())
      {
        line_queue.push(std::move(lines));
        lines = nullptr;
      }
      for (size_t i = 0; i != cfg.parse_thread_num; ++i)
        line_queue.push(nullptr);
      for (auto& th : parse_threads)
        if (th.joinable())
          th.join();
      for (size_t i = 0; i != cfg.train_thread_num; ++i)
        sample_queue.push(nullptr);
      for (auto& th : train_threads)
        if (th.joinable())
          th.join();

      t_end = Time::now();
      cost  = t_end - t_begin;
      spdlog::info("epoch {:4d}, trained on {} samples, costs {:.4f} secs", epoch_i, n_sample,
                   cost.count());

      /*********************************************************
      *  validation                                            *
      *********************************************************/
      if (!cfg.valid_file.empty())
      {
        t_begin = Time::now();
        Parser      parser_valid(cfg.valid_file);
        vector<F>   y_pred;
        vector<int> y_true;
        while (unique_ptr<Sample> sample = Parser::parseSample(parser_valid.nextLine()))
        {
          F pred = model->predict_prob(sample);
          y_pred.emplace_back(pred);
          y_true.emplace_back(sample->y);
          if (cfg.debug)
            spdlog::debug("PRED {:.4f} {}", pred, sample->y);
        }
        t_end = Time::now();
        cost  = t_end - t_begin;
        spdlog::info("{}, {} samples, AUC: {:.6f}, costs {:.4f} secs", cfg.valid_file,
                     y_pred.size(), calc_auc(y_pred, y_true), cost.count());
      }
    }
  }

  /*********************************************************
  *  model saving                                          *
  *********************************************************/
  if (!cfg.save.empty())
  {
    t_begin = Time::now();
    spdlog::info("**************** save model ****************");
    spdlog::info("save to {}", cfg.save);
    model->save(cfg.save);
    t_end = Time::now();
    cost  = t_end - t_begin;
    spdlog::info("finish, costs {:.4f} secs", cost.count());
  }

  /*********************************************************
  *  predict                                               *
  *********************************************************/
  if (!(cfg.test_file.empty() || cfg.test_pred_file.empty()))
  {
    t_begin = Time::now();
    spdlog::info("**************** predict ****************");
    spdlog::info("input: {}", cfg.test_file);
    spdlog::info("output: {}", cfg.test_pred_file);
    Parser   parser_test(cfg.test_file);
    ofstream ofs;
    ofs.open(cfg.test_pred_file, ofstream::out);
    while (unique_ptr<Sample> sample = Parser::parseSample(parser_test.nextLine()))
    {
      F pred = model->predict_prob(sample);
      ofs << pred << endl;
    }
    ofs.close();
    t_end = Time::now();
    cost  = t_end - t_begin;
    spdlog::info("finish, costs {:.4f} secs", cost.count());
  }
  return 0;
}

int check_args()
{
  if (cfg.model != "lr" && cfg.model != "fm")
  {
    cerr << "model must be lr or fm\n";
    return -1;
  }
  if (cfg.seed != -1 && !(cfg.train_thread_num == 1 && cfg.parse_thread_num == 1))
  {
    cerr << "random seed should be used with 1 train_thread and 1 parse_thread\n";
    return -1;
  }
  return 0;
}

int main(int argc, char* argv[])
{
  /*********************************************************
  *  parse program args                                    *
  *********************************************************/
  cxxopts::Options options(argv[0], "\nTraining toolkit for LR/FM on sparse data.\n");
  options.set_tab_expansion().set_width(150);
  string group;
  options.add_option(group, "m", "model", "lr or fm",
                     cxxopts::value<std::string>()->default_value("lr"), "");
  options.add_option(group, "", "train", "training file",
                     cxxopts::value<std::string>()->default_value("../dataset/train.txt"), "");
  options.add_option(group, "", "valid", "validation file",
                     cxxopts::value<std::string>()->default_value("../dataset/valid.txt"), "");
  options.add_option(group, "", "test", "testing file",
                     cxxopts::value<std::string>()->default_value("../dataset/test.txt"), "");
  options.add_option(group, "", "test_pred", "file to save predictions of testing file",
                     cxxopts::value<std::string>()->default_value("../data_output/test_pred.txt"), "");
  options.add_option(group, "i", "load", "file to load model",
                     cxxopts::value<std::string>()->default_value(""), "");
  options.add_option(group, "o", "save", "file to save model",
                     cxxopts::value<std::string>()->default_value("../data_output/model.txt"), "");
  options.add_option(group, "", "w_lr", "learning_rate for linear part",
                     cxxopts::value<F>()->default_value("0.1"), "");
  options.add_option(group, "", "v_lr", "learning_rate for embedding part",
                     cxxopts::value<F>()->default_value("0.1"), "");
  options.add_option(group, "", "w_l2", "l2 regularization for linear part",
                     cxxopts::value<F>()->default_value("0"), "");
  options.add_option(group, "", "v_l2", "l2 regularization for embedding part",
                     cxxopts::value<F>()->default_value("0"), "");
  options.add_option(group, "", "v_stddev", "stddev for embedding initialization",
                     cxxopts::value<F>()->default_value("0.001"), "");
  options.add_option(group, "e", "epoch", "num of epochs",
                     cxxopts::value<uint32_t>()->default_value("10"), "");
  options.add_option(group, "b", "batch_size", "batch_size for mini-batch sgd",
                     cxxopts::value<uint32_t>()->default_value("64"), "");
  options.add_option(group, "k", "factor", "dim of embedding",
                     cxxopts::value<uint32_t>()->default_value("4"), "");
  options.add_option(group, "", "tt", "train thread num",
                     cxxopts::value<uint32_t>()->default_value("10"), "");
  options.add_option(group, "", "pt", "parse thread num",
                     cxxopts::value<uint32_t>()->default_value("3"), "");
  options.add_option(group, "", "seed", "random seed, use with 1 train_thread and 1 parse_thread， -1: no seed",
                     cxxopts::value<long>()->default_value("-1"), "");
  options.add_option(group, "d", "debug", "debug",
                     cxxopts::value<bool>()->default_value("false"), "");
  options.add_options()("h,help", "printing this help message");

  try
  {
    auto args = options.parse(argc, argv);
    if (argc == 1 || args.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }

    cfg.model            = args["model"].as<string>();
    cfg.train_file       = args["train"].as<string>();
    cfg.valid_file       = args["valid"].as<string>();
    cfg.test_file        = args["test"].as<string>();
    cfg.test_pred_file   = args["test_pred"].as<string>();
    cfg.load             = args["load"].as<string>();
    cfg.save             = args["save"].as<string>();
    cfg.w_lr             = args["w_lr"].as<F>();
    cfg.v_lr             = args["v_lr"].as<F>();
    cfg.w_l2             = args["w_l2"].as<F>();
    cfg.v_l2             = args["v_l2"].as<F>();
    cfg.v_stddev         = args["v_stddev"].as<F>();
    cfg.epoch            = args["epoch"].as<uint32_t>();
    cfg.batch_size       = args["batch_size"].as<uint32_t>();
    cfg.k                = args["factor"].as<uint32_t>();
    cfg.train_thread_num = args["tt"].as<uint32_t>();
    cfg.parse_thread_num = args["pt"].as<uint32_t>();
    cfg.seed             = args["seed"].as<long>();
    cfg.debug            = args["debug"].as<bool>();
  } catch (cxxopts::exceptions::exception& exception)
  {
    cerr << "error parsing args: " << exception.what() << std::endl;
    exit(-1);
  }

  if (check_args())
  {
    exit(-1);
  }

  spdlog::info("**************** config ****************\n" + cfg.str());

  /*********************************************************
  *  logger settings                                       *
  *********************************************************/
  spdlog::level::level_enum log_level = cfg.debug ? spdlog::level::debug : spdlog::level::info;
  spdlog::set_level(log_level);

  return run();
}
