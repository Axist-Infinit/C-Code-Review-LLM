/* sample: primevul_train_2442_fixed */
int LibarchivePlugin::extractionFlags() const
{
    return ARCHIVE_EXTRACT_TIME
           | ARCHIVE_EXTRACT_SECURE_NODOTDOT
           | ARCHIVE_EXTRACT_SECURE_SYMLINKS;
}
