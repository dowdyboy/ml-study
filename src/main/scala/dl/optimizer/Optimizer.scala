package dl.optimizer

import breeze.linalg.{DenseMatrix, DenseVector}

trait Optimizer {

  def update(layerIdx:Int,weightMat:DenseMatrix[Double],dWeightMat:DenseMatrix[Double]):DenseMatrix[Double]

  def update(layerIdx:Int,offsetVec:DenseVector[Double],dOffsetVec:DenseVector[Double]):DenseVector[Double]

}
