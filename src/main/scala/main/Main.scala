package main

import breeze.linalg.{Axis, DenseMatrix, DenseVector, max, sum, tile}
import breeze.numerics.exp
import com.typesafe.scalalogging.Logger
import dl.network.FullJoinNetwork


object Main {

  def main(args: Array[String]): Unit = {
    dl.Instances.testMLPNetwork
  }
}
