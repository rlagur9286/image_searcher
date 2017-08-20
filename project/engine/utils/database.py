import logging
import pymysql
import timeit

logging.basicConfig(
    format="[%(name)s][%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()],
    level=logging.INFO
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class DBManager(object):
    def __init__(self):
        """DB Connection Class for Smart audio_diary (SD) Database

        """
        try:
            self.conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='1234',
                                        db='image_search_engine_db', charset='utf8', use_unicode=True)
            self.connected = True
        except pymysql.Error:
            self.connected = False


class ImageManager(DBManager):
    def __init__(self):
        DBManager.__init__(self)

    def retrieve_info_all(self):  # for retrieving Account
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_retrieve_url = "SELECT * FROM product_tb"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve_url)
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_info_all() - Execution Time : %s", stop - start)
                return cur.fetchall()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_info_all()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def retrieve_info_by_PRODUCT_CD(self, PRODUCT_CD):  # for retrieving Account
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_retrieve_url = "SELECT * FROM product_tb WHERE PRODUCT_CD = %s"
        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_retrieve_url, PRODUCT_CD)
                stop = timeit.default_timer()
                logger.debug("DB : retrieve_info_by_PRODUCT_CD() - Execution Time : %s", stop - start)
                return cur.fetchone()

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At retrieve_info_by_PRODUCT_CD()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def insert_product2db(self, PRODUCT_CD, GOODS_NAME, GOODS_IMAGE_URL, BRAND, MODEL, CATEGORY, PRICE):
        start = timeit.default_timer()
        assert self.connected
        query_for_insert = "INSERT INTO product_tb " \
                                "(PRODUCT_CD, GOODS_NAME, GOODS_IMAGE_URL, BRAND, MODEL, CATEGORY, PRICE) " \
                                "VALUES (%s, %s, %s, %s, %s, %s, %s)"

        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                cur.execute(query_for_insert, (PRODUCT_CD, GOODS_NAME, GOODS_IMAGE_URL, BRAND, MODEL, CATEGORY, PRICE))
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : insert_product2db() - Execution Time : %s", stop - start)
            return True

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At insert_product2db()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg

    def delete_info_from_db(self, PRODUCT_CD):
        # START : for calculating execution time
        start = timeit.default_timer()
        assert self.connected  # Connection Check Flag
        query_for_delete = "DELETE FROM product_tb WHERE PRODUCT_CD = %s"

        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cur:
                affected_rows = cur.execute(query_for_delete, PRODUCT_CD)
                self.conn.commit()
                # END : for calculating execution time
                stop = timeit.default_timer()
                logger.debug("DB : delete_info_from_db() - Execution Time : %s", stop - start)
                if affected_rows == 1:
                    return True
                else:
                    return False

        except Exception as exp:
            logger.error(">>>MYSQL ERROR<<<")
            logger.error("At delete_info_from_db()")
            num, error_msg = exp.args
            logger.error("ERROR NO : %s", num)
            logger.error("ERROR MSG : %s", error_msg)
            return error_msg
