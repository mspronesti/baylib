//
// Created by elle on 09/08/21.
//

#include <gtest/gtest.h>
#include <baylib/parser/net_parser.hpp>
#include <baylib/network/bayesian_utils.hpp>



TEST(parser_test, graph_root){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Coma.xdsl");
    ASSERT_TRUE(net1.is_root("MetastCancer"));
    ASSERT_FALSE(net1.is_root("IncrSerCal"));
    ASSERT_FALSE(net1.is_root("BrainTumor"));
    ASSERT_FALSE(net1.is_root("Coma"));
    ASSERT_FALSE(net1.is_root("SevHeadaches"));
    }

TEST(parser_test, graph_dependency){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Coma.xdsl");

    ASSERT_TRUE(net1.has_dependency("MetastCancer", "IncrSerCal"));
    ASSERT_TRUE(net1.has_dependency("MetastCancer", "BrainTumor"));
    ASSERT_TRUE(net1.has_dependency("IncrSerCal", "Coma"));
    ASSERT_TRUE(net1.has_dependency("BrainTumor", "Coma"));
    ASSERT_TRUE(net1.has_dependency("BrainTumor", "SevHeadaches"));

    ASSERT_FALSE(net1.has_dependency("IncrSerCal", "MetastCancer"));
    ASSERT_FALSE(net1.has_dependency("BrainTumor", "MetastCancer"));
    ASSERT_FALSE(net1.has_dependency("Coma", "IncrSerCal"));
    ASSERT_FALSE(net1.has_dependency("Coma", "BrainTumor"));
    ASSERT_FALSE(net1.has_dependency("SevHeadaches", "BrainTumor"));

    ASSERT_FALSE(net1.has_dependency("MetastCancer", "SevHeadaches"));
    ASSERT_FALSE(net1.has_dependency("MetastCancer", "Coma"));
    ASSERT_FALSE(net1.has_dependency("IncrSerCal", "SevHeadaches"));
    ASSERT_FALSE(net1.has_dependency("BrainTumor", "IncrSerCal"));
}

TEST(parser_test, test_node_states){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Coma.xdsl");
    ASSERT_TRUE(net1["MetastCancer"].has_state("present"));
    ASSERT_TRUE(net1["MetastCancer"].has_state("absent"));
    ASSERT_TRUE(net1["IncrSerCal"].has_state("present"));
    ASSERT_TRUE(net1["IncrSerCal"].has_state("absent"));
    ASSERT_TRUE(net1["BrainTumor"].has_state("present"));
    ASSERT_TRUE(net1["BrainTumor"].has_state("absent"));
    ASSERT_TRUE(net1["SevHeadaches"].has_state("present"));
    ASSERT_TRUE(net1["SevHeadaches"].has_state("absent"));
    ASSERT_TRUE(net1["Coma"].has_state("present"));
    ASSERT_TRUE(net1["Coma"].has_state("absent"));

    ASSERT_FALSE(net1["MetastCancer"].has_state("maybe"));
    ASSERT_FALSE(net1["Coma"].has_state("maybe"));

}

TEST(parser_test, test_node_filled){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Coma.xdsl");
    ASSERT_TRUE(bn::cpt_filled_out(net1["MetastCancer"]));
    ASSERT_TRUE(bn::cpt_filled_out(net1["IncrSerCal"]));
    ASSERT_TRUE(bn::cpt_filled_out(net1["BrainTumor"]));
    ASSERT_TRUE(bn::cpt_filled_out(net1["SevHeadaches"]));
    ASSERT_TRUE(bn::cpt_filled_out(net1["Coma"]));
}

TEST(parser_test, test_node_cpt_root){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Coma.xdsl");
    bn::condition cond;
    ASSERT_DOUBLE_EQ(net1["MetastCancer"].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net1["MetastCancer"].table().at(cond)[1], .8);
}

TEST(parser_test, test_node_cpt_children){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FComa.xdsl
    auto net1 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Coma.xdsl");
    bn::condition cond;
    cond.add("MetastCancer", 0);
    ASSERT_DOUBLE_EQ(net1["BrainTumor"].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net1["BrainTumor"].table().at(cond)[1], .8);
    ASSERT_DOUBLE_EQ(net1["IncrSerCal"].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1["IncrSerCal"].table().at(cond)[1], .2);

    cond.add("MetastCancer", 1);
    ASSERT_DOUBLE_EQ(net1["BrainTumor"].table().at(cond)[0], .05);
    ASSERT_DOUBLE_EQ(net1["BrainTumor"].table().at(cond)[1], .95);
    ASSERT_DOUBLE_EQ(net1["IncrSerCal"].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net1["IncrSerCal"].table().at(cond)[1], .8);

    cond.clear();
    cond.add("IncrSerCal", 0);
    cond.add("BrainTumor", 0);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[1], .2);
    cond.add("IncrSerCal", 0);
    cond.add("BrainTumor", 1);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[1], .2);
    cond.add("IncrSerCal", 1);
    cond.add("BrainTumor", 0);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[1], .2);
    cond.add("IncrSerCal", 1);
    cond.add("BrainTumor", 1);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[0], .05);
    ASSERT_DOUBLE_EQ(net1["Coma"].table().at(cond)[1], .95);
}

TEST(parser_test, test_node_cpt_children2){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FVentureBN.xdsl
    auto net2 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/VentureBN.xdsl");
    bn::condition cond;
    cond.add("Success", 0);
    ASSERT_DOUBLE_EQ(net2["Forecast"].table().at(cond)[0], .4);
    ASSERT_DOUBLE_EQ(net2["Forecast"].table().at(cond)[1], .4);
    ASSERT_DOUBLE_EQ(net2["Forecast"].table().at(cond)[2], .2);

    cond.add("Success", 1);
    ASSERT_DOUBLE_EQ(net2["Forecast"].table().at(cond)[0], .1);
    ASSERT_DOUBLE_EQ(net2["Forecast"].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net2["Forecast"].table().at(cond)[2], .6);
}
TEST(parser_test, test_node_cpt_children3){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FVentureBNExpanded.xdsl
    auto net3 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/VentureBNExpanded.xdsl");
    bn::condition cond;
    cond.add("Economy", 0);
    cond.add("Success", 0);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[1], .2);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[2], .1);

    cond.add("Economy", 0);
    cond.add("Success", 1);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[2], .5);

    cond.add("Economy", 1);
    cond.add("Success", 0);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[0], .6);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[2], .1);

    cond.add("Economy", 1);
    cond.add("Success", 1);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[0], .1);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[2], .6);

    cond.add("Economy", 2);
    cond.add("Success", 0);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[0], .5);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[1], .3);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[2], .2);

    cond.add("Economy", 2);
    cond.add("Success", 1);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[1], .4);
    ASSERT_DOUBLE_EQ(net3["Forecast"].table().at(cond)[2], .4);
}

TEST(parser_test, test_node_cpt_children4){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FAnimals.xdsl
    auto net4 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Animals.xdsl");
    bn::condition cond;

    cond.add("Animal", 0);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[1], 1);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[2], 0);

    cond.add("Animal", 1);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[1], .5);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[2], .5);

    cond.add("Animal", 2);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[1], 0);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[2], 1);

    cond.add("Animal", 3);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[0], .5);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[1], .5);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[2], 0);

    cond.add("Animal", 4);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[0], 0);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[1], .5);
    ASSERT_DOUBLE_EQ(net4["Environment"].table().at(cond)[2], .5);
}

TEST(parser_test, test_node_cpt_children5) {
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Credit.xdsl");
    bn::condition cond;
    cond.add("Reliability", 0);
    cond.add("RatioDebInc", 0);
    cond.add("FutureIncome", 0);
    cond.add("Age", 0);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .9);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .1);

    cond.add("Reliability", 0);
    cond.add("RatioDebInc", 0);
    cond.add("FutureIncome", 0);
    cond.add("Age", 2);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .2);

    cond.add("Reliability", 0);
    cond.add("RatioDebInc", 0);
    cond.add("FutureIncome", 1);
    cond.add("Age", 0);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .3);

    cond.add("Reliability", 0);
    cond.add("RatioDebInc", 0);
    cond.add("FutureIncome", 1);
    cond.add("Age", 1);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .8);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .2);

    cond.add("Reliability", 0);
    cond.add("RatioDebInc", 0);
    cond.add("FutureIncome", 1);
    cond.add("Age", 2);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .6);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .4);

    cond.add("Reliability", 0);
    cond.add("RatioDebInc", 1);
    cond.add("FutureIncome", 0);
    cond.add("Age", 0);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .3);

    cond.add("Reliability", 0);
    cond.add("RatioDebInc", 1);
    cond.add("FutureIncome", 0);
    cond.add("Age", 2);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .7);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .3);

    cond.add("Reliability", 1);
    cond.add("RatioDebInc", 0);
    cond.add("FutureIncome", 1);
    cond.add("Age", 2);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .2);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .8);

    cond.add("Reliability", 1);
    cond.add("RatioDebInc", 1);
    cond.add("FutureIncome", 1);
    cond.add("Age", 2);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[0], .001);
    ASSERT_DOUBLE_EQ(net5["CreditWorthiness"].table().at(cond)[1], .999);

}

TEST(parser_test, test_cow){
    //https://repo.bayesfusion.com/network/permalink?net=Small+BNs%2FCredit.xdsl
    auto net5 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Credit.xdsl");
    //https://repo.bayesfusion.com/network/permalink?net=Large+BNs%2FLink.xdsl
    auto net6 = bn::net_parser<double>().load_from_xdsl("../../examples/xdsl/Link.xdsl");
    bn::condition c1;
    std::cout << sizeof(net6);
    const auto& e1 = net5["Income"].table();
    const auto& e2 = net5["Assets"].table();

    ASSERT_EQ(std::addressof(e1[c1][0]), std::addressof(e2[c1][0]));

    bn::condition c2;
    bn::condition c3;
    c2.add("N58_d_f", 0);
    c2.add("N58_d_m", 0);
    c3.add("N57_d_f", 0);
    c3.add("N57_d_m", 0);
    const auto& e3 = net6["N58_d_g"].table();
    const auto& e4 = net6["N57_d_g"].table();
    ASSERT_EQ(std::addressof(e3[c2][0]), std::addressof(e4[c3][0]));
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
