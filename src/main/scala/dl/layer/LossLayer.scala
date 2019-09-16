package dl.layer

import breeze.linalg._
import breeze.numerics.log

trait LossLayer {

  protected def cross_entropy_error(resultVec:DenseVector[Double],oneHotLabelVec:DenseVector[Double]) = {
    val deltaVec:DenseVector[Double] = DenseVector.tabulate(resultVec.length){x => 0.0000001}
    0 - sum(log(resultVec + deltaVec) * oneHotLabelVec)
  }
  protected def cross_entropy_error(resultMat:DenseMatrix[Double],oneHotLabelMat:DenseMatrix[Double]) = {
    val deltaMat:DenseMatrix[Double] = DenseMatrix.tabulate(resultMat.rows,resultMat.cols){(x,y) => 0.0000001}
    sum(log(resultMat + deltaMat) *:* oneHotLabelMat,Axis._1).map(0 - _)
  }

  def forward(dataMat:DenseMatrix[Double],labelMat:DenseMatrix[Double]):Double

  def backward(dout:Double):DenseMatrix[Double]

}
