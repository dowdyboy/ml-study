package dl.layer
import breeze.linalg._
import breeze.numerics._
import com.typesafe.scalalogging.Logger
import dl.common.DLog

class SigmoidLayer extends Layer {

  val logger = Logger(this.getClass)

  private var outMat:DenseMatrix[Double] =  null

  override def forward(inMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    DLog.logMat(logger,inMat)
    val out = sigmoid(inMat)
    outMat = out.copy
    out
  }

  override def backward(dinMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    dinMat *:* outMat *:* (DenseMatrix.tabulate(outMat.rows,outMat.cols){(x,y)=>1.0} - outMat)
  }

}
