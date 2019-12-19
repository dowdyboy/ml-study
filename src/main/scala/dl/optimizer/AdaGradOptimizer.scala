package dl.optimizer
import breeze.linalg.{DenseMatrix, DenseVector}

class AdaGradOptimizer(val learnRate:Double) extends Optimizer {

  import scala.collection.mutable.Map

  var hWeightMap:Map[Int,DenseMatrix[Double]] = Map()
  var hOffsetMap:Map[Int,DenseVector[Double]] = Map()

  override def update(layerIdx: Int, weightMat: DenseMatrix[Double], dWeightMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    var hWeight = hWeightMap.get(layerIdx).getOrElse(DenseMatrix.zeros[Double](weightMat.rows,weightMat.cols))
    val learnRateMat = DenseMatrix.tabulate[Double](dWeightMat.rows,dWeightMat.cols){(x,y)=>learnRate}
    val smallValueMat = DenseMatrix.tabulate[Double](hWeight.rows,hWeight.cols){(x,y)=>0.0000001}
    hWeight = hWeight + (dWeightMat *:* dWeightMat)
    hWeightMap.update(layerIdx,hWeight)
    weightMat - ((learnRateMat *:* dWeightMat) /:/ (hWeight.map(x=>Math.sqrt(x)) + smallValueMat))
  }

  override def update(layerIdx: Int, offsetVec: DenseVector[Double], dOffsetVec: DenseVector[Double]): DenseVector[Double] = {
    var hOffset = hOffsetMap.get(layerIdx).getOrElse(DenseVector.zeros[Double](offsetVec.size))
    hOffset = hOffset + (dOffsetVec * dOffsetVec)
    hOffsetMap.update(layerIdx,hOffset)
    offsetVec - ((learnRate * dOffsetVec) / (hOffset.map(x=>Math.sqrt(x)) + 0.0000001))
  }

}
