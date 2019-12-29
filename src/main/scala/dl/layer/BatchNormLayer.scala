package dl.layer
import breeze.linalg._

class BatchNormLayer(val gamma:Double = 1.0,val beta:Double = 0.0,val momentum:Double = 0.9) extends Layer {

  var gammaMat:DenseMatrix[Double] = null
  var betaMat:DenseMatrix[Double] = null
  var savedMeanMat:DenseMatrix[Double] = null
  var savedVarMat:DenseMatrix[Double] = null
  var batchSize:Int = 0
  var savedCutMeanMat:DenseMatrix[Double] = null
  var savedXnMat:DenseMatrix[Double] = null
  var savedStdMat:DenseMatrix[Double] = null
  var dGammaMat:DenseMatrix[Double] = null
  var dBetaMat:DenseMatrix[Double] = null


  override def forward(inMat: DenseMatrix[Double], isTrain: Boolean): DenseMatrix[Double] = {
    if(gammaMat == null || betaMat == null){
      //使用matrix必须批大小刚好整除数据集大小
      gammaMat = DenseMatrix.tabulate[Double](inMat.rows,inMat.cols){(x,y)=>gamma}
      betaMat = DenseMatrix.tabulate[Double](inMat.rows,inMat.cols){(x,y)=>beta}
    }
    if(savedMeanMat == null || savedVarMat == null){
      savedMeanMat = DenseMatrix.zeros[Double](inMat.rows,inMat.cols)
      savedVarMat = DenseMatrix.zeros[Double](inMat.rows,inMat.cols)
    }
    if(isTrain){
      val meanMat = tile(sum(inMat,Axis._0) / inMat.rows.toDouble,1,inMat.rows)
      val inMatCutMeanMat = inMat - meanMat
      val varMat = tile(sum(inMatCutMeanMat.map(x=>x*x),Axis._0),1,inMatCutMeanMat.rows)
      val stdMat = (varMat + 0.0000001).map(x=>Math.sqrt(x))
      val xnMat = inMatCutMeanMat /:/ stdMat
      batchSize = inMat.rows
      savedCutMeanMat = inMatCutMeanMat
      savedXnMat = xnMat
      savedStdMat = stdMat
      savedMeanMat = (momentum * savedMeanMat) + ((1-momentum) * meanMat)
      savedVarMat = (momentum * savedVarMat) + ((1-momentum) * varMat)
      (gammaMat *:* xnMat) + betaMat
    }else{
      val inMatCutMeanMat = inMat - tile(savedMeanMat(0,::),1,inMat.rows)
      val xnMat = inMatCutMeanMat / (tile(savedVarMat(0,::),1,inMatCutMeanMat.rows) + 0.0000001).map(x=>Math.sqrt(x))
      (tile(gammaMat(0,::),1,xnMat.rows) *:* xnMat) + tile(betaMat(0,::),1,xnMat.rows)
    }
  }

  override def backward(dinMat: DenseMatrix[Double]): DenseMatrix[Double] = {
    val dbeta = tile(sum(dinMat,Axis._0),1,dinMat.rows)
    val dgamma = tile(sum(savedXnMat *:* dinMat,Axis._0),1,dinMat.rows)
    val dXnMat = gammaMat *:* dinMat
    var dXcMat = dXnMat / savedStdMat
    val dStdMat = tile(sum((dXnMat *:* savedCutMeanMat) / (savedStdMat *:* savedStdMat),Axis._0),1,dXnMat.rows).map(x=>0-x)
    val dVarMat = (0.5 * dStdMat) / savedStdMat
    dXcMat = dXcMat + ((2.0 / batchSize) *:* savedCutMeanMat *:* dVarMat)
    val dMuMat = tile(sum(dXcMat,Axis._0),1,dXcMat.rows)
    val dxMat = dXcMat - (dMuMat / batchSize.toDouble)
    dGammaMat = dgamma
    dBetaMat = dbeta
    dxMat
  }
}
