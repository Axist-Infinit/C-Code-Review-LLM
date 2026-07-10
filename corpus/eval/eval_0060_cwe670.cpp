/* sample: primevul_train_3452_vuln */
bool EmbFile::save2(FILE *f) {
  int c;

  m_objStr.streamReset();
  while ((c = m_objStr.streamGetChar()) != EOF) {
    fputc(c, f);
  }
  return true;
}
