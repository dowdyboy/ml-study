package dl.layer

import breeze.linalg.DenseMatrix

trait Layer {

  def forward(inMat:DenseMatrix[Double]):DenseMatrix[Double]

  def backward(dinMat:DenseMatrix[Double]):DenseMatrix[Double]

}
