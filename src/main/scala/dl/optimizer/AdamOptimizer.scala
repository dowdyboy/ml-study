package dl.optimizer
import breeze.linalg.{DenseMatrix, DenseVector}

class AdamOptimizer(val learnRate:Double,betaA:Double = 0.9,betaB:Double = 0.999) extends Optimizer {

  import scala.collection.mutable.Map

  var iter:Int = 0
  var max:Int = Int.MaxValue
  var mWeightMap:Map[Int,DenseMatrix[Double]] = Map()
  var mOffsetMap:Map[Int,DenseVector[Double]] = Map()
  var vWeightMap:Map[Int,DenseMatrix[Double]] = Map()
  var vOffsetMap:Map[Int,DenseVector[Double]] = Map()

  override def update(layerIdx: Int, weightMat: DenseMatrix[Double], dWeightMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    if(layerIdx <= max){
      iter += 1
      max = layerIdx
    }
    val learnRateTemp = learnRate * Math.sqrt(1.0 - Math.pow(betaB,iter)) / (1.0 - Math.pow(betaA,iter))
    var mWeight = mWeightMap.get(layerIdx).getOrElse(DenseMatrix.zeros[Double](weightMat.rows,weightMat.cols))
    var vWeight = vWeightMap.get(layerIdx).getOrElse(DenseMatrix.zeros[Double](weightMat.rows,weightMat.cols))
    mWeight = mWeight + ((1.0 - betaA) * (dWeightMat - mWeight))
    mWeightMap.update(layerIdx,mWeight)
    vWeight = vWeight + ((1.0 - betaB) * ((dWeightMat *:* dWeightMat) - vWeight))
    vWeightMap.update(layerIdx,vWeight)
    weightMat - ((learnRateTemp * mWeight) /:/ vWeight.map(x=>Math.sqrt(x) + 0.0000001))
  }

  override def update(layerIdx: Int, offsetVec: DenseVector[Double], dOffsetVec: DenseVector[Double]): DenseVector[Double] = {
    val learnRateTemp = learnRate * Math.sqrt(1.0 - Math.pow(betaB,iter)) / (1.0 - Math.pow(betaA,iter))
    var mOffset = mOffsetMap.get(layerIdx).getOrElse(DenseVector.zeros[Double](offsetVec.size))
    var vOffset = vOffsetMap.get(layerIdx).getOrElse(DenseVector.zeros[Double](offsetVec.size))
    mOffset = mOffset + ((1.0 - betaA) * (dOffsetVec - mOffset))
    mOffsetMap.update(layerIdx,mOffset)
    vOffset = vOffset + ((1.0 - betaB) * ((dOffsetVec * dOffsetVec) - vOffset))
    vOffsetMap.update(layerIdx,vOffset)
    offsetVec - ((learnRateTemp * mOffset) / vOffset.map(x=>Math.sqrt(x) + 0.0000001))
  }
}
