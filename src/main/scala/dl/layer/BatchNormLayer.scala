package dl.layer
import breeze.linalg._

class BatchNormLayer(val gamma:Double = 1.0,val beta:Double = 0.0,val momentum:Double = 0.9) extends Layer {

  var gammaMat:DenseMatrix[Double] = null
  var betaMat:DenseMatrix[Double] = null
  var dGammaMat:DenseMatrix[Double] = null
  var dBetaMat:DenseMatrix[Double] = null

  var savedMeanVec:DenseVector[Double] = null
  var savedVarVec:DenseVector[Double] = null
  var savedBatchSize:Int = 0
  var savedCutMeanMat:DenseMatrix[Double] = null
  var savedXnMat:DenseMatrix[Double] = null
  var savedStdMat:DenseMatrix[Double] = null


  override def forward(inMat: DenseMatrix[Double], isTrain: Boolean): DenseMatrix[Double] = {
    if(gammaMat == null || betaMat == null){
      gammaMat = DenseMatrix.tabulate[Double](inMat.rows,inMat.cols){(x,y)=>gamma}
      betaMat = DenseMatrix.tabulate[Double](inMat.rows,inMat.cols){(x,y)=>beta}
    }
    if(savedMeanVec == null || savedVarVec == null){
      savedMeanVec = DenseVector.zeros[Double](inMat.cols)
      savedVarVec = DenseVector.zeros[Double](inMat.cols)
    }
    if(isTrain){
      val meanVec = (sum(inMat,Axis._0) / inMat.rows.toDouble).inner
      val inMatCutMeanMat = inMat - tile(meanVec.t,1,inMat.rows)
      val varVec = sum(inMatCutMeanMat.map(x=>x*x),Axis._0).inner
      val stdMat = (tile(varVec.t,1,inMatCutMeanMat.rows) + 0.0000001).map(x=>Math.sqrt(x))
      val xnMat = inMatCutMeanMat /:/ stdMat
      savedBatchSize = inMat.rows
      savedCutMeanMat = inMatCutMeanMat
      savedXnMat = xnMat
      savedStdMat = stdMat
      savedMeanVec = (momentum * savedMeanVec) + ((1-momentum) * meanVec)
      savedVarVec = (momentum * savedVarVec) + ((1-momentum) * varVec)
      (tile(gammaMat(0,::),1,xnMat.rows) *:* xnMat) + tile(betaMat(0,::),1,xnMat.rows)
    }else{
      val inMatCutMeanMat = inMat - tile(savedMeanVec.t,1,inMat.rows)
      val xnMat = inMatCutMeanMat / (tile(savedVarVec.t,1,inMatCutMeanMat.rows) + 0.0000001).map(x=>Math.sqrt(x))
      (tile(gammaMat(0,::),1,xnMat.rows) *:* xnMat) + tile(betaMat(0,::),1,xnMat.rows)
    }
  }

  override def backward(dinMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dbeta = tile(sum(dinMat,Axis._0),1,betaMat.rows)
    val dgamma = tile(sum(savedXnMat *:* dinMat,Axis._0),1,gammaMat.rows)
    val dXnMat = tile(gammaMat(0,::),1,dinMat.rows) *:* dinMat
    var dXcMat = dXnMat / savedStdMat
    val dStdMat = tile(sum((dXnMat *:* savedCutMeanMat) / (savedStdMat *:* savedStdMat),Axis._0),1,dXnMat.rows).map(x=>0-x)
    val dVarMat = (0.5 * dStdMat) / savedStdMat
    dXcMat = dXcMat + ((2.0 / savedBatchSize) *:* savedCutMeanMat *:* dVarMat)
    val dMuMat = tile(sum(dXcMat,Axis._0),1,dXcMat.rows)
    val dxMat = dXcMat - (dMuMat / savedBatchSize.toDouble)
    dGammaMat = dgamma
    dBetaMat = dbeta
    dxMat
  }
}
