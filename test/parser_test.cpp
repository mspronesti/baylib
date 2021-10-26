//
// Created by elle on 09/08/21.
//

#include <gtest/gtest.h>
#include <baylib/smile_utils/smile_utils.hpp>
#include <baylib/network/bayesian_utils.hpp>



TEST(parser_test, graph_root){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Coma.xdsl");
    auto name_map = baylib::make_name_map(net1);
    ASSERT_TRUE(net1.is_root(name_map["MetastCancer"]));
    ASSERT_TRUE(net1.is_root(name_map["MetastCancer"]));
    ASSERT_FALSE(net1.is_root(name_map["IncrSerCal"]));
    ASSERT_FALSE(net1.is_root(name_map["IncrSerCal"]));
    ASSERT_FALSE(net1.is_root(name_map["BrainTumor"]));
    ASSERT_FALSE(net1.is_root(name_map["BrainTumor"]));
    ASSERT_FALSE(net1.is_root(name_map["Coma"]));
    ASSERT_FALSE(net1.is_root(name_map["Coma"]));
    ASSERT_FALSE(net1.is_root(name_map["SevHeadaches"]));
    ASSERT_FALSE(net1.is_root(name_map["SevHeadaches"]));
    }

TEST(parser_test, graph_dependency){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Coma.xdsl");
    auto name_map = baylib::make_name_map(net1);
    ASSERT_TRUE(net1.has_dependency(name_map["MetastCancer"], name_map["IncrSerCal"]));
    ASSERT_TRUE(net1.has_dependency(name_map["MetastCancer"], name_map["BrainTumor"]));
    ASSERT_TRUE(net1.has_dependency(name_map["IncrSerCal"], name_map["Coma"]));
    ASSERT_TRUE(net1.has_dependency(name_map["BrainTumor"], name_map["Coma"]));
    ASSERT_TRUE(net1.has_dependency(name_map["BrainTumor"], name_map["SevHeadaches"]));

    ASSERT_FALSE(net1.has_dependency(name_map["IncrSerCal"], name_map["MetastCancer"]));
    ASSERT_FALSE(net1.has_dependency(name_map["BrainTumor"], name_map["MetastCancer"]));
    ASSERT_FALSE(net1.has_dependency(name_map["Coma"], name_map["IncrSerCal"]));
    ASSERT_FALSE(net1.has_dependency(name_map["Coma"], name_map["BrainTumor"]));
    ASSERT_FALSE(net1.has_dependency(name_map["SevHeadaches"], name_map["BrainTumor"]));

    ASSERT_FALSE(net1.has_dependency(name_map["MetastCancer"], name_map["SevHeadaches"]));
    ASSERT_FALSE(net1.has_dependency(name_map["MetastCancer"], name_map["Coma"]));
    ASSERT_FALSE(net1.has_dependency(name_map["IncrSerCal"], name_map["SevHeadaches"]));
    ASSERT_FALSE(net1.has_dependency(name_map["BrainTumor"], name_map["IncrSerCal"]));
}

TEST(parser_test, test_node_states){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Coma.xdsl");
    auto name_map = baylib::make_name_map(net1);
    ASSERT_TRUE(net1[name_map["MetastCancer"]].has_state("present"));
    ASSERT_TRUE(net1[name_map["MetastCancer"]].has_state("absent"));
    ASSERT_TRUE(net1[name_map["IncrSerCal"]].has_state("present"));
    ASSERT_TRUE(net1[name_map["IncrSerCal"]].has_state("absent"));
    ASSERT_TRUE(net1[name_map["BrainTumor"]].has_state("present"));
    ASSERT_TRUE(net1[name_map["BrainTumor"]].has_state("absent"));
    ASSERT_TRUE(net1[name_map["SevHeadaches"]].has_state("present"));
    ASSERT_TRUE(net1[name_map["SevHeadaches"]].has_state("absent"));
    ASSERT_TRUE(net1[name_map["Coma"]].has_state("present"));
    ASSERT_TRUE(net1[name_map["Coma"]].has_state("absent"));
    ASSERT_FALSE(net1[name_map["MetastCancer"]].has_state("maybe"));
    ASSERT_FALSE(net1[name_map["Coma"]].has_state("maybe"));

}

TEST(parser_test, test_node_filled){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Coma.xdsl");
    auto name_map = baylib::make_name_map(net1);
    ASSERT_TRUE(baylib::cpt_filled_out(net1, name_map["MetastCancer"]));
    ASSERT_TRUE(baylib::cpt_filled_out(net1, name_map["IncrSerCal"]));
    ASSERT_TRUE(baylib::cpt_filled_out(net1, name_map["BrainTumor"]));
    ASSERT_TRUE(baylib::cpt_filled_out(net1, name_map["SevHeadaches"]));
    ASSERT_TRUE(baylib::cpt_filled_out(net1, name_map["Coma"]));
}

TEST(parser_test, test_node_cpt_root){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Coma.xdsl");
    auto name_map = baylib::make_name_map(net1);
    baylib::condition cond;
    ASSERT_DOUBLE_EQ(net1[name_map["MetastCancer"]].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net1[name_map["MetastCancer"]].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net1[name_map["MetastCancer"]].table().at(cond)[1], .8);
    ASSERT_DOUBLE_EQ(net1[name_map["MetastCancer"]].table().at(cond)[1], .8);
}

TEST(parser_test, test_node_cpt_children){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Coma.xdsl");
    auto name_map = baylib::make_name_map(net1);
    baylib::condition cond;
    cond.add(name_map["MetastCancer"], 0);
    ASSERT_DOUBLE_EQ(net1[name_map["BrainTumor"]].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net1[name_map["BrainTumor"]].table().at(cond)[1], .8);
    ASSERT_DOUBLE_EQ(net1[name_map["IncrSerCal"]].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1[name_map["IncrSerCal"]].table().at(cond)[1], .2);

    cond.add(name_map["MetastCancer"], 1);
    ASSERT_DOUBLE_EQ(net1[name_map["BrainTumor"]].table().at(cond)[0], .05);
    ASSERT_DOUBLE_EQ(net1[name_map["BrainTumor"]].table().at(cond)[1], .95);
    ASSERT_DOUBLE_EQ(net1[name_map["IncrSerCal"]].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net1[name_map["IncrSerCal"]].table().at(cond)[1], .8);

    cond.clear();
    cond.add(name_map["IncrSerCal"], 0);
    cond.add(name_map["BrainTumor"], 0);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[1], .2);
    cond.add(name_map["IncrSerCal"], 0);
    cond.add(name_map["BrainTumor"], 1);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[1], .2);
    cond.add(name_map["IncrSerCal"], 1);
    cond.add(name_map["BrainTumor"], 0);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[1], .2);
    cond.add(name_map["IncrSerCal"], 1);
    cond.add(name_map["BrainTumor"], 1);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[0], .05);
    ASSERT_DOUBLE_EQ(net1[name_map["Coma"]].table().at(cond)[1], .95);
}

TEST(parser_test, test_node_cpt_children2){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FVentureBN.xdsl
    auto net2 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/VentureBN.xdsl");
    auto name_map = baylib::make_name_map(net2);
    baylib::condition cond;
    cond.add(name_map["Success"], 0);
    ASSERT_DOUBLE_EQ(net2[name_map["Forecast"]].table().at(cond)[0], .4);
    ASSERT_DOUBLE_EQ(net2[name_map["Forecast"]].table().at(cond)[1], .4);
    ASSERT_DOUBLE_EQ(net2[name_map["Forecast"]].table().at(cond)[2], .2);

    cond.add(name_map["Success"], 1);
    ASSERT_DOUBLE_EQ(net2[name_map["Forecast"]].table().at(cond)[0], .1);
    ASSERT_DOUBLE_EQ(net2[name_map["Forecast"]].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net2[name_map["Forecast"]].table().at(cond)[2], .6);
}
TEST(parser_test, test_node_cpt_children3){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FVentureBNExpanded.xdsl
    auto net3 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/VentureBNExpanded.xdsl");
    auto name_map = baylib::make_name_map(net3);
    baylib::condition cond;
    cond.add(name_map["Economy"], 0);
    cond.add(name_map["Success"], 0);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[1], .2);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[2], .1);

    cond.add(name_map["Economy"], 0);
    cond.add(name_map["Success"], 1);
    auto temp = net3[name_map["Forecast"]].table();
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[2], .5);

    cond.add(name_map["Economy"], 1);
    cond.add(name_map["Success"], 0);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[0], .6);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[2], .1);

    cond.add(name_map["Economy"], 1);
    cond.add(name_map["Success"], 1);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[0], .1);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[2], .6);

    cond.add(name_map["Economy"], 2);
    cond.add(name_map["Success"], 0);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[0], .5);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[2], .2);

    cond.add(name_map["Economy"], 2);
    cond.add(name_map["Success"], 1);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[1], .4);
    ASSERT_DOUBLE_EQ(net3[name_map["Forecast"]].table().at(cond)[2], .4);
}

TEST(parser_test, test_node_cpt_children4){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FAnimals.xdsl
    auto net4 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Animals.xdsl");
    auto name_map = baylib::make_name_map(net4);
    baylib::condition cond;

    cond.add(name_map["Animal"], 0);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[1], 1);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[2], 0);

    cond.add(name_map["Animal"], 1);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[1], .5);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[2], .5);

    cond.add(name_map["Animal"], 2);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[1], 0);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[2], 1);

    cond.add(name_map["Animal"], 3);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[0], .5);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[1], .5);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[2], 0);

    cond.add(name_map["Animal"], 4);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[1], .5);
    ASSERT_DOUBLE_EQ(net4[name_map["Environment"]].table().at(cond)[2], .5);
}

TEST(parser_test, test_node_cpt_children5) {
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = baylib::xdsl_parser<double>().deserialize("../../examples/xdsl/Credit.xdsl");
    auto name_map = baylib::make_name_map(net5);
    baylib::condition cond;
    cond.add(name_map["Reliability"], 0);
    cond.add(name_map["RatioDebInc"], 0);
    cond.add(name_map["FutureIncome"], 0);
    cond.add(name_map["Age"], 0);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .9);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .1);

    cond.add(name_map["Reliability"], 0);
    cond.add(name_map["RatioDebInc"], 0);
    cond.add(name_map["FutureIncome"], 0);
    cond.add(name_map["Age"], 2);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .2);

    cond.add(name_map["Reliability"], 0);
    cond.add(name_map["RatioDebInc"], 0);
    cond.add(name_map["FutureIncome"], 1);
    cond.add(name_map["Age"], 0);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .3);

    cond.add(name_map["Reliability"], 0);
    cond.add(name_map["RatioDebInc"], 0);
    cond.add(name_map["FutureIncome"], 1);
    cond.add(name_map["Age"], 1);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .2);

    cond.add(name_map["Reliability"], 0);
    cond.add(name_map["RatioDebInc"], 0);
    cond.add(name_map["FutureIncome"], 1);
    cond.add(name_map["Age"], 2);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .6);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .4);

    cond.add(name_map["Reliability"], 0);
    cond.add(name_map["RatioDebInc"], 1);
    cond.add(name_map["FutureIncome"], 0);
    cond.add(name_map["Age"], 0);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .3);

    cond.add(name_map["Reliability"], 0);
    cond.add(name_map["RatioDebInc"], 1);
    cond.add(name_map["FutureIncome"], 0);
    cond.add(name_map["Age"], 2);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .3);

    cond.add(name_map["Reliability"], 1);
    cond.add(name_map["RatioDebInc"], 0);
    cond.add(name_map["FutureIncome"], 1);
    cond.add(name_map["Age"], 2);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .8);

    cond.add(name_map["Reliability"], 1);
    cond.add(name_map["RatioDebInc"], 1);
    cond.add(name_map["FutureIncome"], 1);
    cond.add(name_map["Age"], 2);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[0], .001);
    ASSERT_DOUBLE_EQ(net5[name_map["CreditWorthiness"]].table().at(cond)[1], .999);

}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
