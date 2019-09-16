package dl.layer
import breeze.linalg._
import breeze.numerics._
import com.typesafe.scalalogging.Logger
import dl.common.DLog

class SoftmaxLossLayer extends LossLayer {

  val logger = Logger(this.getClass)

  private var softmaxResultMat:DenseMatrix[Double] = null
  private var labelMat:DenseMatrix[Double] = null

  private def softmax(in:DenseVector[Double]) = {
    val maxInValue:Double = max(in)
    val expIn:DenseVector[Double] = exp(in.map(x=>x-maxInValue))
    val sumExpIn:Double = sum(expIn)
    expIn.map(x=>x/sumExpIn)
  }
  private def softmax(in:DenseMatrix[Double]) = {
    val maxInValue:DenseVector[Double] = max(in,Axis._1)
    val expIn = exp(in - tile(maxInValue,1,in.cols))
    val sumExpIn = tile(sum(expIn,Axis._1),1,in.cols)
    expIn /:/ sumExpIn
  }

  override def forward(dataMat: DenseMatrix[Double], labelMat: DenseMatrix[Double]):Double = {
    DLog.logMat(logger,dataMat)
    softmaxResultMat = softmax(dataMat)
    DLog.logMat(logger,softmaxResultMat)
    this.labelMat = labelMat.copy
    val errorVec = cross_entropy_error(softmaxResultMat,labelMat)
    sum(errorVec) / errorVec.length.toDouble
  }

  override def backward(dout: Double): DenseMatrix[Double] = {
    (softmaxResultMat - labelMat) /:/ DenseMatrix.tabulate(labelMat.rows,labelMat.cols){(x,y) => labelMat.rows.toDouble}
  }

}
