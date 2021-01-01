import tensorflow as tf
from bpemb import BPEmb

bpemb_bn = BPEmb(lang="bn", vs=VOCAB_SIZE, dim=VOCAB_DIM)


def summarize(input_document):
    summarized = evaluate(input_document=input_document)[0].numpy()
    summarized = np.expand_dims(summarized[1:], 0)
    return bpemb_bn.decode_ids(summarized[0])


def evaluate(input_document):
    input_document = [bpemb_bn.encode_ids(input_document)]
    input_document = pad_sequences(
        input_document, maxlen=TEXT_LENGTH, padding='post', truncating='post')

    encoder_input = tf.expand_dims(input_document[0], 0)

    decoder_input = [START_TOKEN]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(SUMMARY_LENGTH):
        predictions, attention_weights = transformer(
            (encoder_input, output),
            training=False,
        )

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == END_TOKEN:
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights

TEST_SUMMARY = "ট্রাকের ধাক্কায় সড়কে ছিটকে পড়ে প্রবাসীর মৃত্যু"
TEST_CONTENT = "যশোরের বাঘারপাড়া উপজেলায় ট্রাকের ধাক্কায় এক প্রবাসীর মৃত্যু হয়েছে। তাঁর নাম আবু সাঈদ (৪০)। রোববার রাত আটটার দিকে উপজেলার খাজুরায় বাঘারপাড়া-কালীগঞ্জ সড়কে আরিফ ব্রিকসের সামনে এ দুর্ঘটনা ঘটে।"\
" নিহত আবু সাঈদ বাঘারপাড়া উপজেলার বন্দবিলা ইউনিয়নের দাঁতপুর গ্রামের সদর আলী দফাদারের ছেলে। তিনি মালয়েশিয়াপ্রবাসী ছিলেন। প্রত্যক্ষদর্শীর বরাত দিয়ে পুলিশ জানায়, রোববার রাতে খাজুরা বাজার থেকে বাঘারপাড়া-কালীগঞ্জ সড়ক দিয়ে"\
" বাইসাইকেলে করে বাড়ি ফিরছিলেন আবু সাঈদ। রাত আটটার দিকে তিনি আরিফ ব্রিকসের সামনে পৌঁছান। এ সময় পেছন দিক থেকে আসা বাঘারপাড়াগামী একটি দ্রুতগামী ট্রাক সাইকেলটিকে ধাক্কা দেয়। বাইসাইকেল থেকে সড়কের ওপর ছিটকে পড়ে"\
" সেখানেই সাঈদের মৃত্যু হয়। আবু সাঈদের ভাই রেজাউল দফাদার বলেন, আবু সাঈদ মালয়েশিয়ায় চাকরি করতেন। সম্প্রতি ছুটিতে তিনি দেশে এসেছিলেন।"\
" বাঘারপাড়া থানা ভারপ্রাপ্ত কর্মকর্তা (ওসি) সৈয়দ আল মামুন বলেন, ট্রাকের ধাক্কায় আবু সাঈদ নামের এক বাইসাইকেলচালক নিহত হয়েছেন।"