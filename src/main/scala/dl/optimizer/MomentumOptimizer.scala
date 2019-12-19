package dl.optimizer
import breeze.linalg.{DenseMatrix, DenseVector}

class MomentumOptimizer(val learnRate:Double,val momentum:Double) extends Optimizer {

  import scala.collection.mutable.Map

  var vWeightMap:Map[Int,DenseMatrix[Double]] = Map()
  var vOffsetMap:Map[Int,DenseVector[Double]] = Map()

  override def update(layerIdx:Int,weightMat: DenseMatrix[Double], dWeightMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    var vMatrix = vWeightMap.get(layerIdx).getOrElse(DenseMatrix.zeros[Double](weightMat.rows,weightMat.cols))
    val momentumMat = DenseMatrix.tabulate[Double](vMatrix.rows,vMatrix.cols){(x,y)=>momentum}
    val learnRateMat = DenseMatrix.tabulate[Double](dWeightMat.rows,dWeightMat.cols){(x,y)=>learnRate}
    vMatrix = (momentumMat *:* vMatrix) - (learnRateMat *:* dWeightMat)
    vWeightMap.update(layerIdx,vMatrix)
    weightMat + vMatrix
  }

  override def update(layerIdx:Int,offsetVec: DenseVector[Double], dOffsetVec: DenseVector[Double]): DenseVector[Double] = {
    var vOffset = vOffsetMap.get(layerIdx).getOrElse(DenseVector.zeros[Double](offsetVec.size))
    vOffset = (momentum * vOffset) - (learnRate * dOffsetVec)
    vOffsetMap.update(layerIdx,vOffset)
    offsetVec + vOffset
  }

}
