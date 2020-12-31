import time


def format_time_passed(start):
    end = time.time() - start
    days = end // 86400
    end -= days * 86400
    hours = end // 3600
    end -= hours * 3600
    minutes = end // 60
    end -= minutes * 60
    seconds = end
    return '{:02} day {:02} hr {:02} min {:02} sec'.format(
        int(days), int(hours), int(minutes), int(seconds)
    )


def print_header(file_name):
    header = "{}\t\t{}\t\t{}\t\t{}\t\t{}"
    sep = "=" * 150
    epoch_str = '{:<5}'.format("EPOCH")
    batch_str = '{:<5}'.format("BATCH")
    loss_str = '{:<10}'.format("LOSS")
    time_passed_str = '{:<20}'.format("TIME ELAPSED")
    summary_str = "{:<50}".format("SUMMARY")

    with open(file_name, "a") as fp:
        print(sep, file=fp)
        print(header.format(epoch_str, batch_str, loss_str,
                            time_passed_str, summary_str), file=fp)
        print(sep, file=fp)


def print_info(epoch, batch, loss, summary, file_name, start):
    row = "{}\t\t{}\t\t{}\t\t{}\t{}"
    sep = "-" * 150
    epoch_str = '{:5}'.format(epoch)
    batch_str = '{:5}'.format(batch)
    loss_str = '{:<2.7f}'.format(loss)
    time_passed_str = format_time_passed(start)

    with open(file_name, "a") as fp:
        print(row.format(epoch_str, batch_str, loss_str,
                         time_passed_str, summary), file=fp)
        print(sep, file=fp)


def save_score(path, epoch, batch, loss):
    with open(path, "a") as fp:
        fp.write(f"{epoch},{batch},{loss}\n")
