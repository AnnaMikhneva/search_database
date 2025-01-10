CREATE TABLE User (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    age INT NOT NULL,
    education_field VARCHAR(255) NOT NULL
);

CREATE TABLE Query (
    query_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    query_text VARCHAR(500) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
);