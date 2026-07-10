/* sample: primevul_train_2095_vuln */
int64_t OutputFile::readImpl(char* /*buffer*/, int64_t /*length*/) {
  raise_warning("cannot read from a php://output stream");
  return -1;
}
