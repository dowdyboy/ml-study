package dl.common

import breeze.linalg.DenseMatrix
import com.typesafe.scalalogging.Logger

object DLog {

  def logMat(logger:Logger,mat:DenseMatrix[Double]) = {
    val newMat = mat.map(_.formatted("%.2f"))
    val sb = new StringBuilder
    (0 until newMat.rows).foreach{i=>
      (0 until newMat.cols).foreach{k=>
        sb.append(newMat.valueAt(i,k) + "\t")
      }
      sb.append("\n")
    }
    logger.info(sb.toString())
  }

}
