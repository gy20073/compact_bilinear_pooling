/**
Since Bidmach doesn't support kernel SVM, stop exploring

call command
:load /home/yang/code/fv_layer/yang_exps/ft_classify/train_test_bidmach.scala
*/

val infile:String = "/home/yang/exp_data/fv_layer/exp_yang_fv/activations_CUB_VGG-M_14.mat"
val a:DMat = load(infile,"trainY")

val feature=rand(20, 20);
val (nn, nopts) = GLM.SVMlearner(datamat, labelmat)
