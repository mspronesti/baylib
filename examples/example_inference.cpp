#include <baylib/smile_utils/smile_utils.hpp>
#include <baylib/inference/gibbs_sampling.hpp>
#include <baylib/inference/logic_sampling.hpp>
#include <baylib/inference/likelihood_weighting.hpp>
#include <iostream>

// Evidences in a bayesian network are a way to include posteriori knowledge
// about the state of the network and analyze how this knowledge affects the
// rest of the network

int main(int argc, char** argv){
    using namespace baylib;
    using namespace baylib::inference;

    // Evidences can be applied to all types of networks, we use Credit as an example
    // https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    xdsl_parser<double> parser;
    auto network = parser.deserialize("../../examples/xdsl/Credit.xdsl");

    // Here we declare the Gibbs sampling algorithm to perform approximate inference
    // PLEASE NOTICE: Gibbs Sampling fails with bayesian networks with deterministic nodes
    //                (it computes wrong marginal probabilities, it's a well known theoretical
    //                limit of this sampling approach) hence it should not be used in such cases
    baylib::inference::gibbs_sampling  gibbs_sampler(network, 10000, 4);

    // getting the output only takes passing the bayesian network to the make_inference
    // method. Every algorithm has this method
    std::cout << gibbs_sampler.make_inference();

    // Let's now assume we know, as evidence, the values of "Debit" and "Income"
    // to see how to inference simulation behaves
    //network["Debit"].set_as_evidence(0);
    //network["Income"].set_as_evidence(1);

    // To detect if a node was set as evidence you can use the is_evidence and evidence_state methods
    //if(network["Debit"].is_evidence())
    //    std::cout << network["Debit"].evidence_state() << '\n';

    // The algorithm will automatically detect all evidences set and use them in the inferences
    std::cout << gibbs_sampler.make_inference() << '\n';

    // To clear the evidences use the clear_evidence on the desired nodes or use the util clear_network_evidences
    // to clean all the network from evidences
    //network["Debit"].clear_evidence();
    //clear_network_evidences(network);

    // The network now should be on its base state
    // Let's know try with a different algorithm to see if the results are coherent
    // with gibbs sampling: likelihood weighting
    likelihood_weighting likely_weigh(network, 10000, 4);

    std::cout << likely_weigh.make_inference() << '\n';

    // Different evidences can now be specified
    // let's see how likelihood-weighting behaves
    //network["Debit"].set_as_evidence(1);
    //network["Income"].set_as_evidence(0);

    std::cout << likely_weigh.make_inference() << '\n';
    return 0;
}