package dl.layer
import breeze.linalg._
import com.typesafe.scalalogging.Logger

class ReluLayer extends Layer {

  val logger = Logger(this.getClass)

  private var maskMat:DenseMatrix[Double] = null

  override def forward(inMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    maskMat = inMat.map(x=>{if(x>0) 1.0 else 0.0})
    inMat.map(x=>{if(x>0) x else 0.0})
  }

  override def backward(dinMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    dinMat *:* maskMat
  }

}
