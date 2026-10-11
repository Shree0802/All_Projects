# 📚 Online Bookstore Data Analysis Using SQL

## 📌 Project Overview

The **Online Bookstore Data Analysis** project uses SQL to manage and analyze an online bookstore's data. It focuses on books, customers, and orders to extract meaningful business insights using SQL queries.

This project demonstrates practical SQL skills, including database creation, data querying, filtering, aggregation, joins, grouping, and inventory analysis.

The project is suitable for beginners learning SQL and aspiring Data Analysts who want to build practical projects for their portfolios.

## 🎯 Project Objectives

* Create and manage a relational database for an online bookstore.
* Store and organize books, customers, and order information.
* Analyze book prices, genres, publication years, and stock availability.
* Identify customer purchasing patterns.
* Calculate total sales revenue and customer spending.
* Analyze book sales by genre and author.
* Identify frequently ordered books.
* Calculate remaining book inventory after fulfilling orders.

## 🛠️ Tools & Technologies

* **SQL:** Database querying and data analysis
* **PostgreSQL:** Relational database management system
* **CSV:** Dataset storage and data import
* **GitHub:** Project documentation and version control

## 📂 Project Structure

```text
OnlineBookStore/
│
├── Books.csv
├── Customers.csv
├── Orders.csv
├── OnlineBookStore.sql
└── README.md
```

### File Description

| File                  | Description                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------- |
| `Books.csv`           | Contains book information, including titles, authors, genres, prices, publication years, and stock quantities. |
| `Customers.csv`       | Contains customer information, including names, email addresses, phone numbers, cities, and countries.         |
| `Orders.csv`          | Contains order information, including customer IDs, book IDs, order dates, quantities, and total amounts.      |
| `OnlineBookStore.sql` | Contains SQL scripts for database creation, table creation, data import, and basic and advanced queries.       |
| `README.md`           | Documents the project, objectives, technologies, and analysis performed.                                       |

## 📊 Dataset Information

The project includes three datasets:

| Dataset   | Number of Records | Number of Columns |
| --------- | ----------------: | ----------------: |
| Books     |               500 |                 7 |
| Customers |               500 |                 6 |
| Orders    |               500 |                 6 |
| **Total** |         **1,500** |            **19** |

These figures represent the records and columns in the provided CSV files.

### 1. Books Dataset

Contains information about the bookstore's inventory.

**Columns:**

* `Book_ID`: Unique book identifier
* `Title`: Book title
* `Author`: Author's name
* `Genre`: Book category
* `Published_Year`: Year of publication
* `Price`: Book price
* `Stock`: Available stock quantity

### 2. Customers Dataset

Contains customer information.

**Columns:**

* `Customer_ID`: Unique customer identifier
* `Name`: Customer's name
* `Email`: Customer's email address
* `Phone`: Customer's phone number
* `City`: Customer's city
* `Country`: Customer's country

### 3. Orders Dataset

Contains information about customer orders.

**Columns:**

* `Order_ID`: Unique order identifier
* `Customer_ID`: Identifier linking an order to a customer
* `Book_ID`: Identifier linking an order to a book
* `Order_Date`: Date of the order
* `Quantity`: Number of books ordered
* `Total_Amount`: Total amount recorded for the order

## 🗄️ Database Schema

The database consists of three relational tables:

```text
Books
-----
Book_ID (Primary Key)
Title
Author
Genre
Published_Year
Price
Stock

Customers
---------
Customer_ID (Primary Key)
Name
Email
Phone
City
Country

Orders
------
Order_ID (Primary Key)
Customer_ID (Foreign Key)
Book_ID (Foreign Key)
Order_Date
Quantity
Total_Amount
```

### Relationships

* One customer can have multiple orders.
* One book can appear in multiple orders.
* The `Orders` table connects customers and books through foreign keys.

## 🔍 SQL Analysis Performed

The project includes basic and advanced SQL queries.

### 1. Basic SQL Queries

The basic analysis covers:

* Retrieving all books, customers, and orders.
* Filtering books by genre.
* Finding books published after a specified year.
* Retrieving customers from Canada.
* Filtering orders by date.
* Calculating total available book stock.
* Finding the most expensive book.
* Identifying orders with quantities greater than one.
* Finding orders with total amounts exceeding $20.
* Listing distinct book genres.
* Finding the book with the lowest stock.
* Calculating total revenue from orders.

**SQL concepts used:**
`SELECT`, `WHERE`, `BETWEEN`, `SUM()`, `DISTINCT`, `ORDER BY`, and `LIMIT`.

### 2. Advanced SQL Queries

The advanced analysis includes:

* Calculating total books sold for each genre.
* Finding the average price of books in the Fantasy genre.
* Identifying customers who have placed at least two orders.
* Finding the most frequently ordered book.
* Retrieving the three most expensive Fantasy books.
* Calculating total books sold by each author.
* Identifying cities associated with orders exceeding $30.
* Finding the customer who spent the most.
* Calculating remaining inventory after accounting for recorded orders.

**SQL concepts used:**
`JOIN`, `GROUP BY`, `HAVING`, `COUNT()`, `AVG()`, `SUM()`, `COALESCE()`, and aggregate functions.

## 💡 Business Questions Answered

This project explores questions such as:

1. Which books belong to the Fiction genre?
2. Which books were published after 1950?
3. How many books are available in total?
4. What is the highest-priced book?
5. Which book has the lowest stock?
6. What is the total revenue recorded from orders?
7. Which genres generate the highest book sales by quantity?
8. Which authors have sold the most books?
9. Which customers have placed multiple orders?
10. Which book is ordered most frequently?
11. Who is the highest-spending customer?
12. How much stock remains after accounting for recorded orders?

## 📈 Key Skills Demonstrated

Through this project, I practiced:

* Relational database design
* SQL database and table creation
* CSV data import
* Primary and foreign keys
* Data filtering and sorting
* Aggregate functions
* Grouping and filtering aggregated results
* Inner joins and left joins
* Substantive business-question analysis
* Sales and inventory analysis
* Writing structured SQL queries

## ⚙️ How to Run the Project

### Step 1: Install PostgreSQL

Install PostgreSQL and open a SQL client such as pgAdmin or connect through the PostgreSQL command-line interface.

### Step 2: Download the Repository

Clone the repository or download the project files from GitHub.

### Step 3: Create the Database

Execute the database creation statement from `OnlineBookStore.sql`:

```sql
CREATE DATABASE OnlineBookstore;
```

Connect to the newly created database before creating the tables.

### Step 4: Create the Tables

Run the table creation statements in `OnlineBookStore.sql` to create the `Books`, `Customers`, and `Orders` tables.

### Step 5: Import the CSV Files

Import the following datasets into their corresponding tables:

* `Books.csv` → `Books`
* `Customers.csv` → `Customers`
* `Orders.csv` → `Orders`

**Note:** Update the local file paths in the SQL script before importing the data. The existing script contains a Windows file path specific to the original computer.

Alternatively, use pgAdmin's Import/Export Data feature to import each CSV file. Ensure that the column names and data types match the table definitions.

### Step 6: Execute the Analysis Queries

Run the basic and advanced queries in `OnlineBookStore.sql` to explore the bookstore data and answer the business questions.

## 🚀 Future Improvements

Possible improvements include:

* Creating a Power BI dashboard using the SQL analysis results.
* Developing monthly and yearly sales reports.
* Analyzing revenue by genre and author.
* Identifying top-selling books and high-value customers.
* Adding customer purchase trend analysis.
* Improving inventory management through low-stock alerts.
* Optimizing queries and adding appropriate indexes.

## 👨‍💻 Author

**Prathamesh B. Patil**

Interested in Data Analytics, SQL, Python, Power BI, and Machine Learning.

---

⭐ If you find this project useful, consider giving the repository a star!

