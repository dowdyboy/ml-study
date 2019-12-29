package dl.layer
import breeze.linalg._
import breeze.numerics._
import com.typesafe.scalalogging.Logger

class SigmoidLayer extends Layer {

  private var outMat:DenseMatrix[Double] =  null

  override def forward(inMat: DenseMatrix[Double],isTrain:Boolean): DenseMatrix[Double] = {
    val out = sigmoid(inMat)
    outMat = out.copy
    out
  }

  override def backward(dinMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    dinMat *:* (DenseMatrix.tabulate(outMat.rows,outMat.cols){(x,y)=>1.0} - outMat) *:* outMat
  }

}
