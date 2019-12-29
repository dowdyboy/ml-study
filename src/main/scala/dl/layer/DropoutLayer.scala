package dl.layer
import breeze.linalg.DenseMatrix

class DropoutLayer(val ratio:Double = 0.5) extends Layer {

  var maskMat:DenseMatrix[Double] = null

  override def forward(inMat: DenseMatrix[Double], isTrain: Boolean): DenseMatrix[Double] = {
    if(isTrain){
      maskMat = DenseMatrix.rand[Double](inMat.rows,inMat.cols).map(x=>if(x > ratio) 1.0 else 0.0)
      inMat *:* maskMat
    }else{
      inMat * (1.0 - ratio)
    }
  }

  override def backward(dinMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    dinMat *:* maskMat
  }
}
