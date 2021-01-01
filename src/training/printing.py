import time

SEP_LENGTH = 200


def get_elapsed_time(start):
    end = int(time.time() - start)
    days = end // 86400
    end -= days * 86400
    hours = end // 3600
    end -= hours * 3600
    minutes = end // 60
    end -= minutes * 60
    seconds = end
    return f'{days} day {hours} hr {minutes} min {seconds} sec'


def print_header(file_name):
    header = "{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}"
    sep = "=" * SEP_LENGTH
    epoch_str = '{:<5}'.format("EPOCH")
    batch_str = '{:<5}'.format("BATCH")
    loss_str = '{:<10}'.format("LOSS")
    acc_str = '{:<10}'.format("ACC")
    time_passed_str = '{:<20}'.format("TIME ELAPSED")
    summary_str = "{:<50}".format("SUMMARY")

    with open(file_name, "a") as fp:
        print(sep, file=fp)
        print(header.format(epoch_str, batch_str, loss_str, acc_str,
                            time_passed_str, summary_str), file=fp)
        print(sep, file=fp)


def print_info(epoch, batch, loss, acc, summary, file_name, start):
    sep = "-" * SEP_LENGTH
    epoch_str = f'{epoch:5}'
    batch_str = f'{batch:5}'
    loss_str = f'{loss:<2.7f}'
    acc_str = f'{acc:<2.7f}'
    time_passed_str = get_elapsed_time(start)

    with open(file_name, "a") as fp:
        print(f"{epoch_str}\t\t{batch_str}\t\t{loss_str}\t\t{acc_str}\t\t{time_passed_str:<20}\t{summary}", file=fp)
        print(sep, file=fp)


def save_score(path, epoch, batch, loss, accuracy):
    with open(path, "a") as fp:
        fp.write(f"{epoch},{batch},{loss},{accuracy}\n")
