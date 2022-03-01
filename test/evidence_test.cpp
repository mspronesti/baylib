//
// Created by paolo on 03/09/21.
//

#include <gtest/gtest.h>
#include <baylib/network/bayesian_net.hpp>
#include <baylib/smile_utils/smile_utils.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>
#include <baylib/inference/rejection_sampling.hpp>
#include <baylib/network/bayesian_utils.hpp>

#ifdef BAYLIB_CUDA
#include "baylib/inference/cuda/logic_sampling_cuda.hpp"
#include "baylib/inference/cuda/likelihood_weighting_cuda.hpp"
#endif

#ifdef BAYLIB_OPENCL
#include "baylib/inference/opencl/logic_sampling_opencl.hpp"
#include <baylib/inference/opencl/adaptive_importance_sampling_opencl.hpp>
#endif

#define THREADS std::thread::hardware_concurrency()
#define SAMPLES 10000
#define MEMORY 500*(std::pow(2,30))
#define TOLERANCE 0.06

using namespace baylib::inference;
using probability_type = double;
template<class Variable>
using bnet = baylib::bayesian_net<Variable>;




template<typename Probability, class Variable>
        std::vector<baylib::marginal_distribution<Probability>> get_results(const bnet<Variable> &bn){
            std::vector<baylib::marginal_distribution<Probability>> results{
                    gibbs_sampling<bnet<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                    likelihood_weighting<bnet<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                    rejection_sampling<bnet<Variable>>(bn, SAMPLES, THREADS).make_inference(),
#ifdef BAYLIB_CUDA
                    logic_sampling_cuda<bnet<Variable>>(bn, SAMPLES).make_inference(),
                    likelihood_weighting_cuda<bnet<Variable>>(bn, SAMPLES).make_inference(),
#endif
#ifdef BAYLIB_OPENCL
                    logic_sampling_opencl<bnet<Variable>>(bn, SAMPLES, MEMORY).make_inference(),
                    adaptive_importance_sampling_opencl<bnet<Variable>>(bn, SAMPLES, MEMORY).make_inference(),
#endif
            };
            return results;
        }

template<typename Probability, class Variable>
        std::vector<baylib::marginal_distribution<Probability>> get_results_heavy(const bnet<Variable> &bn){
            std::vector<baylib::marginal_distribution<Probability>> results{
                    likelihood_weighting<bnet<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                    gibbs_sampling<bnet<Variable>>(bn, SAMPLES, THREADS).make_inference(),
                    rejection_sampling<bnet<Variable>>(bn, SAMPLES, THREADS).make_inference(),
#ifdef BAYLIB_OPENCL
                    adaptive_importance_sampling_opencl<bnet<Variable>>(bn, SAMPLES, MEMORY).make_inference(),
#endif
#ifdef BAYLIB_CUDA
                    likelihood_weighting_cuda<bnet<Variable>>(bn, SAMPLES).make_inference()
#endif
            };
            return results;
        }

TEST(evidence_test, evidence_noisy){
    auto net1 = baylib::xdsl_parser<probability_type>().deserialize("../../examples/xdsl/VentureBN.xdsl");
    auto name_map = baylib::make_name_map(net1);
    net1[name_map["Forecast"]].set_as_evidence(0);
    for (baylib::marginal_distribution<double>& result: get_results<double>(net1)){
        ASSERT_NEAR(result[name_map["Success"]][0], .5, TOLERANCE);
        ASSERT_NEAR(result[name_map["Success"]][1], .5, TOLERANCE);
        ASSERT_NEAR(result[name_map["Forecast"]][0], 1, TOLERANCE);
    }
}

TEST(evidence_test, evidence_coma){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = baylib::xdsl_parser<probability_type>().deserialize("../../examples/xdsl/Coma.xdsl");
    auto name_map = baylib::make_name_map(net1);
    net1[name_map["Coma"]].set_as_evidence(0);

    for(baylib::marginal_distribution<double>& result: get_results<double>(net1)){

        ASSERT_NEAR(result[name_map["MetastCancer"]][0], .43, TOLERANCE);
        ASSERT_NEAR(result[name_map["MetastCancer"]][1], .58, TOLERANCE);

        ASSERT_NEAR(result[name_map["IncrSerCal"]][0], .8, TOLERANCE);
        ASSERT_NEAR(result[name_map["IncrSerCal"]][1], .2, TOLERANCE);

        ASSERT_NEAR(result[name_map["Coma"]][0], 1, TOLERANCE);
        ASSERT_NEAR(result[name_map["Coma"]][1], 0, TOLERANCE);

        ASSERT_NEAR(result[name_map["BrainTumor"]][0], .2, TOLERANCE);
        ASSERT_NEAR(result[name_map["BrainTumor"]][1], .8, TOLERANCE);

        ASSERT_NEAR(result[name_map["SevHeadaches"]][0], .64, TOLERANCE);
        ASSERT_NEAR(result[name_map["SevHeadaches"]][1], .36, TOLERANCE);

    }

    net1[name_map["Coma"]].clear_evidence();
    net1[name_map["IncrSerCal"]].set_as_evidence(0);
    net1[name_map["SevHeadaches"]].set_as_evidence(1);

    for(baylib::marginal_distribution<double>& result: get_results<double>(net1)){

        ASSERT_NEAR(result[name_map["MetastCancer"]][0], .48, TOLERANCE);
        ASSERT_NEAR(result[name_map["MetastCancer"]][1], .52, TOLERANCE);

        ASSERT_NEAR(result[name_map["IncrSerCal"]][0], 1, TOLERANCE);
        ASSERT_NEAR(result[name_map["IncrSerCal"]][1], 0, TOLERANCE);

        ASSERT_NEAR(result[name_map["Coma"]][0], .8, TOLERANCE);
        ASSERT_NEAR(result[name_map["Coma"]][1], .2, TOLERANCE);

        ASSERT_NEAR(result[name_map["BrainTumor"]][0], .07, TOLERANCE);
        ASSERT_NEAR(result[name_map["BrainTumor"]][1], .93, TOLERANCE);

        ASSERT_NEAR(result[name_map["SevHeadaches"]][0], 0, TOLERANCE);
        ASSERT_NEAR(result[name_map["SevHeadaches"]][1], 1, TOLERANCE);
    }

    baylib::clear_network_evidences(net1);
    net1[name_map["SevHeadaches"]].set_as_evidence(1);

    for(baylib::marginal_distribution<double>& result: get_results<double>(net1)){

        ASSERT_NEAR(result[name_map["MetastCancer"]][0], .19, TOLERANCE);
        ASSERT_NEAR(result[name_map["MetastCancer"]][1], .81, TOLERANCE);

        ASSERT_NEAR(result[name_map["IncrSerCal"]][0], .31, TOLERANCE);
        ASSERT_NEAR(result[name_map["IncrSerCal"]][1], .69, TOLERANCE);

        ASSERT_NEAR(result[name_map["Coma"]][0], .30, TOLERANCE);
        ASSERT_NEAR(result[name_map["Coma"]][1], .70, TOLERANCE);

        ASSERT_NEAR(result[name_map["BrainTumor"]][0], .04, TOLERANCE);
        ASSERT_NEAR(result[name_map["BrainTumor"]][1], .96, TOLERANCE);

        ASSERT_NEAR(result[name_map["SevHeadaches"]][0], 0, TOLERANCE);
        ASSERT_NEAR(result[name_map["SevHeadaches"]][1], 1, TOLERANCE);
    }

}

// Barley is quite a heavy network with very big CPTs,
// hence  this test might take some time
TEST(evidence_test, evidence_barley){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FBarleyWeed.xdsl
    auto net = baylib::xdsl_parser<probability_type>().deserialize("../../examples/xdsl/BarleyWeed.xdsl");
    auto name_map = baylib::make_name_map(net);
    net[name_map["frspsum"]].set_as_evidence(0);
    int i = 0;
    for(baylib::marginal_distribution<double>& result: get_results_heavy<double>(net)){
        ASSERT_NEAR(result[name_map["udbr"]][0], .16, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][1], .03, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][2], .07, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][3], .09, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][4], .14, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][5], .18, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][6], .21, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][7], .10, TOLERANCE);
        ASSERT_NEAR(result[name_map["udbr"]][8], .01, TOLERANCE);
    }
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}