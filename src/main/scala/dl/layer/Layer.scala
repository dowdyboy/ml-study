package dl.layer

import breeze.linalg.DenseMatrix

trait Layer {

  def forward(inMat:DenseMatrix[Double],isTrain:Boolean):DenseMatrix[Double]

  def backward(dinMat:DenseMatrix[Double]):DenseMatrix[Double]

}
