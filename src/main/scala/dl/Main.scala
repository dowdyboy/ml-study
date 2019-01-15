package dl

import dl.basic.Perceptron

object Main {

  def main(args: Array[String]): Unit = {
    println(Perceptron.xor(0,0))
    println(Perceptron.xor(0,1))
    println(Perceptron.xor(1,0))
    println(Perceptron.xor(1,1))
  }
}
