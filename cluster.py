import numpy as np

from numpy import ndarray, array, random
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from queue import PriorityQueue


class Cluster:
    def __init__(self):
        self.documents_: dict[int, ndarray] = {}
        self.composite_: ndarray | None = None
    
    def __iter__(self):
        items = self.documents_.items()
        for item in items:
            yield item
    
    def __getitem__(self, id: int) -> ndarray:
        assert id in self.documents_
        return self.documents_[id]
    
    def __lt__(self, _) -> bool:
        return False
    
    def __len__(self) -> int:
        return len(self.documents_)
    
    def add_document(self, id: int, feature: list[float] | ndarray) -> None:
        if isinstance(feature, list):
            feature: ndarray = array(feature, dtype=float)
        if self.composite_ is None:
            self.composite_: ndarray = feature
        else:
            self.composite_ += feature
        self.documents_[id] = feature
        
    def remove_document(self, id: int) -> None:
        if id in self.documents_:
            self.composite_ -= self.documents_[id]
            del self.documents_[id]
    
    @property
    def composite_vector(self) -> float | ndarray:
        if self.composite_ is None:
            return 0.
        return self.composite_
    
    @property
    def documents(self) -> dict[int, ndarray]:
        return self.documents_


class KMeans:
    @staticmethod
    def choose_smartely(ndocs: int, docs: ndarray | list[tuple[int, ndarray]]) -> list[ndarray]:
        """选取初始质心

        Args:
            ndocs (int): 质心数量
            docs (ndarray | list[tuple[int, ndarray]]): 数据点

        Returns:
            list[list[float]] | ndarray | list[ndarray]: 输出
        """
        length: int = len(docs)
        if isinstance(docs, list):
            docs = array([doc for _, doc in docs])
        chosen_docs: list[ndarray | None] = [None] * ndocs
        chosen_doc: ndarray = docs[random.choice(length)]
        chosen_docs[0] = chosen_doc
        closest: ndarray = 1. - docs @ chosen_doc
        potential: float = closest.sum()
        
        # 选取剩余的质心
        for count in range(1, ndocs):
            randval: float = random.random() * potential
            for index in range(length):
                dist: float = closest[index]
                if randval <= dist:
                    break
                randval -= dist
            chosen_docs[count] = docs[index]
        
        return chosen_docs
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 300):
        assert n_clusters > 1 and max_iter > 0
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
    
    def section(self, X: ndarray | list[tuple[int, ndarray]]) -> list[Cluster]:
        cluster_centers: list[ndarray] = self.cluster_centers_
        sectioned_clusters: list[Cluster] = [Cluster() for _ in range(len(cluster_centers))]
        
        if isinstance(X, ndarray):
            similarity: ndarray = np.einsum('ij,kj->ik', X, self.cluster_centers_)
            max_index: ndarray = np.argmax(similarity, axis=-1)
            for i, index in enumerate(max_index):
                    sectioned_clusters[index].add_document(i, X[i])
        elif isinstance(X, list):
            similarity: ndarray = np.einsum('ij,kj->ik', [x for _, x in X], self.cluster_centers_)
            max_index: ndarray = np.argmax(similarity, axis=-1)
            for i, index in enumerate(max_index):
                sectioned_clusters[index].add_document(X[i][0], X[i][-1])
        else:
            raise TypeError
        
        return sectioned_clusters
    
    def refine_clusters(self, clusters: list[Cluster]) -> float:
        """根据k均值算法迭代优化聚类

        Args:
            clusters (list[Cluster]): 簇

        Returns:
            float: 准则函数的值
        """
        max_iter: int = self.max_iter
        loop_count: int = 0
        changed: bool = True
        
        while changed and loop_count < max_iter:
            changed: bool = False
            
            items: list[tuple[int]] = []
            for cluster_id, cluster in enumerate(clusters):
                for item_id, _ in cluster:
                    items.append((cluster_id, item_id))
        
            for item in items:
                cluster_id_base, item_id = item
                cluster_base: Cluster = clusters[cluster_id_base]
                vector: ndarray = cluster_base[item_id]
                composite_vector_base: ndarray = cluster_base.composite_vector
                norm_base: float = norm(composite_vector_base)
                norm_base_moved: float = norm(composite_vector_base - vector)
                
                eval_max: float = -2.
                max_index: int = 0
                
                for cluster_id_target, cluster_target in enumerate(clusters):
                    if cluster_id_target == cluster_id_base:
                        continue
                    composite_vector_target: ndarray = cluster_target.composite_vector
                    norm_target: float = norm(composite_vector_target)
                    norm_target_moved: float = norm(composite_vector_target + vector)
                    eval_moved: float = norm_base_moved + norm_target_moved - norm_base - norm_target
                    if eval_moved > eval_max:
                        eval_max: float = eval_moved
                        max_index: int = cluster_id_target

                if eval_max > 0.:
                    cluster_base.remove_document(item_id)
                    clusters[max_index].add_document(item_id, vector)
                    changed: bool = True
            loop_count += 1
            
        inertia: float = 0.
        for cluster in clusters:
            inertia -= norm(cluster.composite_vector)
        return inertia
    
    def fit(self, X: list[list[float]] | ndarray | list[ndarray]):
        n_clusters: int = self.n_clusters
        assert n_clusters < len(X)
        length: int = len(X)
        if isinstance(X, list):
            n_features_in: int = len(X[0])
            for i in range(1, length):
                assert len(X[i]) == n_features_in
            self.n_features_in_: int = n_features_in
        else:
            self.n_features_in_: int = len(X[0])
        
        X: ndarray = normalize(X, norm='l1')
        self.cluster_centers_: list[ndarray] = self.choose_smartely(n_clusters, X)
        sectioned_clusters: list[Cluster] = self.section(X)
        self.inertia_: float = self.refine_clusters(sectioned_clusters)
        
        labels: list[int | None] = [None] * length
        for cluster_id, cluster in enumerate(sectioned_clusters):
            for item_id, _ in cluster:
                labels[item_id] = cluster_id
        self.labels_: list[int] = labels
        return self


class BisectingKMeans(KMeans):
    def __init__(self, n_clusters: int, limit_eval: float, max_iter: int = 300):
        assert limit_eval > 0.
        super().__init__(n_clusters, max_iter)
        self.limit_eval: float = limit_eval

    def repeated_bisection(self, nclusters: int, limit_eval: float) -> list[Cluster]:
        """执行重复二分聚类

        Args:
            nclusters (int): 簇的数量
            limit_eval (float): 准则函数增幅阈值

        Returns:
            list[Cluster]: 指定数量的簇（Cluster）构成的集合
        """
        documents: ndarray = self.documents_
        
        cluster: Cluster = Cluster()
        for document_id, feature in enumerate(documents):
            cluster.add_document(document_id, feature)
        
        choose_smartely: function = self.choose_smartely
        self.cluster_centers_: list[ndarray] = choose_smartely(2, documents)
        section: function = self.section
        sectioned_clusters: list[Cluster] = section(documents)
        refine_clusters: function = self.refine_clusters
        sctioned_gain: float = -refine_clusters(sectioned_clusters) - norm(cluster.composite_vector)
        if sctioned_gain < limit_eval:
            return [cluster]
        
        que: PriorityQueue[tuple[float, Cluster, list[Cluster] | None]] = PriorityQueue(nclusters)
        que.put((-sctioned_gain, cluster, sectioned_clusters))
        
        while not que.empty() and not que.full():
            _, cluster, sectioned = que.get()
            if sectioned is None:
                que.put((0., cluster, None))
                break
            for c in sectioned:
                if len(c) < 2:
                    que.put((0., c, None))
                    continue
                documents: list[tuple[int, ndarray]] = list(c.documents.items())
                self.cluster_centers_: list[ndarray] = choose_smartely(2, documents)
                sectioned_cs: list[Cluster] = section(documents)
                sctioned_gain: float = -refine_clusters(sectioned_cs) - norm(c.composite_vector)
                if sctioned_gain > limit_eval:  # 若二分后准则函数的增幅小于阈值的话，此次二分不生效
                    que.put((-sctioned_gain, c, sectioned_cs))
                else:
                    que.put((0., c, None))
        
        clusters: list[Cluster | None] = [None] * que.qsize()
        i: int = 0
        while not que.empty():
            _, cluster, sectioned_clusters = que.get()
            clusters[i] = cluster
            i += 1
        return clusters

    def fit(self, X: list[list[float]] | ndarray | list[ndarray]):
        n_clusters: int = self.n_clusters
        assert n_clusters < len(X)
        length: int = len(X)
        if isinstance(X, list):
            n_features_in: int = len(X[0])
            for i in range(1, length):
                assert len(X[i]) == n_features_in
            self.n_features_in_: int = n_features_in
        else:
            self.n_features_in_: int = len(X[0])
        
        self.documents_: ndarray = normalize(X, norm='l1')
        clusters: list[Cluster] = self.repeated_bisection(n_clusters, self.limit_eval)
        self.n_clusters: int = len(clusters)
        
        labels: list[int | None] = [None] * length
        for cluster_id, cluster in enumerate(clusters):
            for item_id, _ in cluster:
                labels[item_id] = cluster_id
        self.labels_: list[int] = labels
        return self
