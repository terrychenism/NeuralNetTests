template <typename Dtype>
void Net<Dtype>::SetAllNeedBackward()
{
	for (int layer_id = layers_.size() - 1; layer_id >= 0; --layer_id) {
		layer_need_backward_[layer_id] = true;
		for (int bottom_id = 0; bottom_id < bottom_vecs_[layer_id].size();
               ++bottom_id) {
        bottom_need_backward_[layer_id][bottom_id] = true;
        }
	}
}
