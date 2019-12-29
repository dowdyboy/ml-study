import breeze.linalg._
import org.junit.Test

class DemoTest {

  @Test
  def breezeTest(): Unit ={
    val mat = DenseMatrix(
      (1.0,2.0,3.0),
      (4.0,5.0,6.0)
    )
    val meanMat = tile(sum(mat,Axis._0) / mat.rows.toDouble,1,mat.rows)
    println(meanMat)
    println(tile(meanMat(0,::),1,meanMat.rows))
    val matCutMeanMat = mat - meanMat
    val varMat = tile(sum(matCutMeanMat.map(x=>x*x),Axis._0),1,matCutMeanMat.rows)
    val stdMat = (varMat + 0.0000001).map(x=>Math.sqrt(x))
    val xnMat = matCutMeanMat /:/ stdMat
    println(xnMat)
    println(" -- ")
    val vec = DenseVector(1.0,2.0,3.0)
    println(tile(vec.t,1,2))
  }

  @Test
  def breezeRandTest(): Unit ={
    val mat = DenseMatrix.rand[Double](2,3)
    println(mat)
  }

}
