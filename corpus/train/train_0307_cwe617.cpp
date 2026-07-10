/* sample: primevul_train_3518_fixed */
void SoundTouch::setChannels(uint numChannels)
{
    if (!verifyNumberOfChannels(numChannels)) return;

    channels = numChannels;
    pRateTransposer->setChannels((int)numChannels);
    pTDStretch->setChannels((int)numChannels);
}
