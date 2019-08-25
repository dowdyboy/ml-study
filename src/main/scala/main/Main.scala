package main

import breeze.linalg.{Axis, DenseMatrix, DenseVector, max, sum, tile}
import breeze.numerics.exp
import com.typesafe.scalalogging.Logger
import dl.network.FullJoinNetwork


object Main {

  def main(args: Array[String]): Unit = {

    if(args.length != 3){
      println("args error")
    }else{
//      "assets/mnist/t10k-images.idx3-ubyte",
//      "assets/mnist/t10k-labels.idx1-ubyte",
//      "args.txt"
      dl.Instances.testFullJoinNetwork(
        args(0),args(1),args(2)
      )
    }
  }
}
