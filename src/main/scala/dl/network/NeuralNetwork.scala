package dl.network

import breeze.linalg._
import breeze.numerics._


object NeuralNetwork {

  trait ActiveFunction
  trait LossFunction

  case object Sigmoid extends ActiveFunction
  case object Relu extends ActiveFunction
  case object Identity extends ActiveFunction
  case object Softmax extends ActiveFunction
  case object MeanSquaredError extends LossFunction
  case object CrossEntropyError extends LossFunction

  case class PracticeConf(
                          iterNumber:Int,
                          learnRate:Double,
                          batSize:Int,
                          activeFunction:ActiveFunction,
                          outputFunction:ActiveFunction,
                          lossFunction:LossFunction
                        )
  case class PredictConf(activeFunction:ActiveFunction,outputFunction:ActiveFunction)

  trait SeqWrapper[T]
  case class SeqVector[T](seq:Seq[T]) extends SeqWrapper[T]
  case class SeqMatrix[T](seq:Seq[Seq[T]]) extends SeqWrapper[T]
}

class NeuralNetwork {

  protected def softmax(in:DenseVector[Double]) = {
    val maxInValue:Double = max(in)
    val expIn:DenseVector[Double] = exp(in.map(x=>x-maxInValue))
    val sumExpIn:Double = sum(expIn)
    expIn.map(x=>x/sumExpIn)
  }
  protected def softmax(in:DenseMatrix[Double]) = {
    val maxInValue:DenseVector[Double] = max(in,Axis._1)
    val expIn = exp(in - tile(maxInValue,1,in.cols))
    val sumExpIn = tile(sum(expIn,Axis._1),1,in.cols)
    expIn /:/ sumExpIn
  }

  protected def cross_entropy_error(resultVec:DenseVector[Double],oneHotLabelVec:DenseVector[Double]) = {
    0 - sum(log(resultVec) * oneHotLabelVec)
  }
  protected def cross_entropy_error(resultMat:DenseMatrix[Double],oneHotLabelMat:DenseMatrix[Double]) = {
    sum(log(resultMat) *:* oneHotLabelMat,Axis._1).map(0 - _)
  }

  protected def mean_squared_error(resultVec:DenseVector[Double],oneHotLabelVec:DenseVector[Double]) = {
    sum((resultVec - oneHotLabelVec).map(x=>x*x)) / 2
  }
  protected def mean_squared_error(resultMat:DenseMatrix[Double],oneHotLabelMat:DenseMatrix[Double]) = {
    sum((resultMat - oneHotLabelMat).map(x=>x*x),Axis._1).map(_ / 2)
  }

}
