package dl.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}

trait Optimizer {

  def update(weightMat:DenseMatrix[Double],dWeightMat:DenseMatrix[Double],learnRate:Double):DenseMatrix[Double]

  def update(offsetVec:DenseVector[Double],dOffsetVec:DenseVector[Double],learnRate:Double):DenseVector[Double]

}
