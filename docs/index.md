# ##Deep Learning自習室

本サイトは、斎藤康毅著「ゼロから作るDeep Learning」を参考にしながら、Deep Learningを学習していく場です。実際に[Ivory](https://pypi.org/project/ivory/)という名前のライブラリを作成していきます。

参考のために、以下のGitレポジトリをクローンします。

+ [「ゼロから作るDeep Learning」](https://www.oreilly.co.jp/books/9784873117584/) [<i class="fab fa-github"></i>](https://github.com/oreilly-japan/deep-learning-from-scratch)
+ [「ゼロから作るDeep Learning ❷」](https://www.oreilly.co.jp/books/9784873118369/) [<i class="fab fa-github"></i>](https://github.com/oreilly-japan/deep-learning-from-scratch-2)
+ [「PythonとKerasによるディープラーニング」](https://book.mynavi.jp/ec/products/detail/id=90124)  [<i class="fab fa-github"></i>](https://github.com/fchollet/deep-learning-with-python-notebooks)
+ [「Neural Network Libraries」](https://nnabla.org/ja/) [<i class="fab fa-github"></i>](https://github.com/sony/nnabla)
+ [「Neural Network Libraries - Python API Examples」](https://nnabla.readthedocs.io/en/latest/python/examples.html) [<i class="fab fa-github"></i>](https://github.com/sony/nnabla-examples)


```python
from ivory.utils import repository

repository.clone('scratch')
repository.clone('scratch2')
repository.clone('nnabla')
repository.clone('nnabla-examples')
repository.clone('keras')
```

TensorFlowに関しては、[チュートリアル](https://www.tensorflow.org/alpha)を直接参照します。
