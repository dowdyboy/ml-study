package dl.layer
import breeze.linalg._
import com.typesafe.scalalogging.Logger

class AffineLayer(var weightMat:DenseMatrix[Double],var offsetVec:DenseVector[Double]) extends Layer {

  private var xMat:DenseMatrix[Double] = null

  var dWeightMat:DenseMatrix[Double] = null
  var dOffsetVec:DenseVector[Double] = null

  override def forward(inMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    xMat = inMat.copy
    (inMat * weightMat) + tile(offsetVec.t,1,inMat.rows)
  }

  override def backward(dinMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    dWeightMat = xMat.t * dinMat
    dOffsetVec = sum(dinMat,Axis._0).inner
    dinMat * weightMat.t
  }

}
