select * from user;

select * from query;

SELECT * FROM User WHERE age > 25;  -- Пользователи старше 25 лет

SELECT * FROM Query WHERE query_text LIKE 'phonetics';

SELECT user_id, COUNT(*) AS query_count
FROM Query
GROUP BY user_id;

SELECT education_field, AVG(age) AS average_age
FROM User
GROUP BY education_field;





