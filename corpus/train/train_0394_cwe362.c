/* sample: primevul_c_train_2460_fixed */
void put_busid_priv(struct bus_id_priv *bid)
{
	if (bid)
		spin_unlock(&bid->busid_lock);
}
