# 🛍️ ShopSphere
### Modern Full Stack E-Commerce Platform

<p align="center">
  <img src="https://img.shields.io/badge/Project-Full%20Stack%20E--Commerce-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Internship-CodeAlpha-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white" />
  <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white" />
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black" />
  <img src="https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white" />
  <img src="https://img.shields.io/badge/Express.js-000000?style=flat&logo=express&logoColor=white" />
  <img src="https://img.shields.io/badge/MongoDB-47A248?style=flat&logo=mongodb&logoColor=white" />
</p>

<p align="center">
  <b>A feature-rich, responsive, and modern online shopping experience built using HTML, CSS, JavaScript, Node.js, Express.js, and MongoDB.</b>
</p>

---

## 📌 Table of Contents

- [📖 About the Project](#-about-the-project)
- [✨ Key Features](#-key-features)
- [🖥️ Screenshots](#️-screenshots)
- [🛠️ Technology Stack](#️-technology-stack)
- [🏗️ System Architecture](#️-system-architecture)
- [📂 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [🔑 Demo Credentials](#-demo-credentials)
- [📡 API Documentation](#-api-documentation)
- [🔐 Security Features](#-security-features)
- [🧪 Testing Checklist](#-testing-checklist)
- [🚀 Future Enhancements](#-future-enhancements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [👨‍💻 Author](#-author)

---

## 📖 About the Project

**ShopSphere** is a modern, full-stack E-Commerce web application developed as part of the **CodeAlpha Full Stack Development Internship**.

The platform provides a complete online shopping experience, allowing users to browse products, search and filter items, manage shopping carts, place orders, and track their purchases.

Built with a structured **Model-View-Controller (MVC) architecture**, the application combines a Node.js and Express.js backend with a responsive Vanilla JavaScript frontend and MongoDB database.

The interface features a modern glassmorphic design, smooth animations, interactive components, and Light/Dark theme support.

### 🎯 Project Objectives

- Develop a complete full-stack E-Commerce application.
- Implement secure user authentication and authorization.
- Build RESTful APIs using Node.js and Express.js.
- Manage application data using MongoDB and Mongoose.
- Create an interactive and responsive shopping interface.
- Implement product management, cart functionality, and order processing.
- Understand real-world full-stack application architecture.

---

## ✨ Key Features

### 🔐 1. Authentication & User Management

- Secure user registration and login.
- JWT-based authentication.
- Password hashing using Bcrypt.
- HttpOnly cookie-based authentication support.
- Bearer token authentication.
- Protected routes and admin authorization.
- User profile management.
- Update personal information and shipping addresses.
- Password update functionality.

### 📦 2. Product Catalog & Management

- 25+ pre-seeded realistic products.
- Products organized into 7 categories.
- Dynamic product listing.
- Individual product detail pages.
- Product ratings and reviews information.
- Category-based filtering.
- Price range filtering.
- Rating-based filtering.
- Search with live autocomplete suggestions.
- Sorting by:
  - Price: Low to High
  - Price: High to Low
  - Newest Arrivals
  - Highest Rated
- Server-side and client-side pagination.
- Admin product management.

### 🛒 3. Shopping Cart & Wishlist

- Add products to shopping cart.
- Update product quantities.
- Remove products from cart.
- Calculate subtotal dynamically.
- Automatic tax and shipping calculations.
- Guest shopping cart using Local Storage.
- Synchronize guest cart after login.
- Add and remove products from wishlist.
- Persistent cart management for authenticated users.

### 💳 4. Checkout & Order Management

- Multi-step checkout process.
- Shipping address collection.
- Multiple payment method options:
  - Credit/Debit Card
  - PayPal
  - Cash on Delivery (COD)
- Order placement and confirmation.
- Automatic stock management.
- Order history.
- Order tracking.
- Order status management:
  - Processing
  - Shipped
  - Delivered

*Note: Payment options are part of the checkout workflow. Real payment processing requires integration with a payment gateway.*

### 👤 5. User Dashboard

- View personal profile information.
- Update account details.
- Manage shipping address.
- View previous orders.
- Track current orders.
- Access shopping history.

### 🎨 6. Modern UI/UX Design

- Modern glassmorphism interface.
- Responsive product grid layouts.
- Light and Dark theme support.
- Theme preference saved in Local Storage.
- Smooth CSS animations.
- Interactive navigation bar.
- Mobile-friendly navigation menu.
- Toast notification system.
- Interactive product cards.
- Responsive layouts for desktop, tablet, and mobile.

### ⚡ 7. Backend & Database

- MVC-based backend architecture.
- RESTful API development.
- MongoDB database integration.
- Mongoose ODM.
- Input validation.
- Centralized error handling.
- File upload middleware.
- Environment variable configuration.
- Automatic in-memory MongoDB fallback for demo execution.

---

## 🖥️ Screenshots

Add your actual project screenshots in the `screenshots/` folder and update the image paths below.

| Homepage | Product Listing |
|---|---|
| ![Homepage](screenshots/homepage.png) | ![Products](screenshots/products.png) |

| Product Details | Shopping Cart |
|---|---|
| ![Product Details](screenshots/product-details.png) | ![Shopping Cart](screenshots/cart.png) |

| Checkout | User Dashboard |
|---|---|
| ![Checkout](screenshots/checkout.png) | ![Dashboard](screenshots/dashboard.png) |

| Login Page | Dark Mode |
|---|---|
| ![Login](screenshots/login.png) | ![Dark Mode](screenshots/dark-mode.png) |

---

## 🛠️ Technology Stack

### Frontend

| Technology | Purpose |
|---|---|
| HTML5 | Website structure |
| CSS3 | Styling and responsive layouts |
| JavaScript ES6+ | Dynamic functionality |
| CSS Variables | Theme management |
| Fetch API | Backend communication |
| Local Storage | Guest cart and theme persistence |

### Backend

| Technology | Purpose |
|---|---|
| Node.js | JavaScript runtime |
| Express.js | Backend framework |
| MVC Architecture | Application organization |
| REST API | Client-server communication |

### Database

| Technology | Purpose |
|---|---|
| MongoDB | NoSQL database |
| Mongoose | Database modeling and management |
| mongodb-memory-server | In-memory database fallback |

### Security & Utilities

| Package | Purpose |
|---|---|
| jsonwebtoken | Authentication |
| bcryptjs | Password hashing |
| express-validator | Request validation |
| cookie-parser | Cookie management |
| cors | Cross-origin resource sharing |
| multer | File uploads |
| dotenv | Environment configuration |

---

## 🏗️ System Architecture

ShopSphere follows a client-server architecture with an MVC-based backend.

```text
              ┌──────────────────────┐
              │       USER           │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   FRONTEND CLIENT    │
              │                      │
              │ HTML5, CSS3, JS ES6+ │
              └──────────┬───────────┘
                         │
                    REST API
                         │
                         ▼
              ┌──────────────────────┐
              │    EXPRESS SERVER    │
              │                      │
              │ Authentication      │
              │ Middleware           │
              │ Route Management     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │     CONTROLLERS      │
              │                      │
              │ Auth Controller      │
              │ Product Controller   │
              │ Cart Controller      │
              │ Order Controller     │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │       MODELS         │
              │                      │
              │ User                 │
              │ Product              │
              │ Cart                 │
              │ Order                │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │      MONGODB         │
              │                      │
              │ Database Storage     │
              └──────────────────────┘
```

### Application Workflow

1. Users interact with the frontend.
2. Frontend sends requests to Express.js REST APIs.
3. Middleware validates requests and authenticates users.
4. Controllers process application logic.
5. Mongoose models communicate with MongoDB.
6. Backend returns JSON responses.
7. Frontend dynamically updates the interface.

---

## 📂 Project Structure

```text
shopsphere/
│
├── client/
│   │
│   ├── css/
│   │   ├── main.css
│   │   ├── components.css
│   │   ├── dark-mode.css
│   │   └── responsive.css
│   │
│   ├── js/
│   │   ├── api.js
│   │   ├── auth.js
│   │   ├── cart.js
│   │   ├── wishlist.js
│   │   ├── products.js
│   │   ├── product-details.js
│   │   ├── checkout.js
│   │   ├── profile.js
│   │   ├── toast.js
│   │   └── main.js
│   │
│   ├── pages/
│   │   ├── products.html
│   │   ├── product-details.html
│   │   ├── cart.html
│   │   ├── checkout.html
│   │   ├── login.html
│   │   ├── register.html
│   │   ├── profile.html
│   │   ├── order-success.html
│   │   ├── order-history.html
│   │   ├── about.html
│   │   ├── contact.html
│   │   └── 404.html
│   │
│   └── index.html
│
├── server/
│   │
│   ├── config/
│   │   └── db.js
│   │
│   ├── controllers/
│   │   ├── authController.js
│   │   ├── productController.js
│   │   ├── cartController.js
│   │   └── orderController.js
│   │
│   ├── middleware/
│   │   ├── authMiddleware.js
│   │   ├── errorMiddleware.js
│   │   ├── uploadMiddleware.js
│   │   └── validateMiddleware.js
│   │
│   ├── models/
│   │   ├── User.js
│   │   ├── Product.js
│   │   ├── Cart.js
│   │   └── Order.js
│   │
│   ├── routes/
│   │   ├── authRoutes.js
│   │   ├── productRoutes.js
│   │   ├── cartRoutes.js
│   │   └── orderRoutes.js
│   │
│   ├── seeders/
│   │   └── seedData.js
│   │
│   ├── utils/
│   │   ├── apiFeatures.js
│   │   └── generateToken.js
│   │
│   ├── .env.example
│   └── server.js
│
├── package.json
├── package-lock.json
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

Follow these instructions to run ShopSphere locally.

### Prerequisites

Make sure you have installed:

- [Node.js](https://nodejs.org/) v18.0.0 or higher
- [npm](https://www.npmjs.com/) v9.0.0 or higher
- [MongoDB Community Server](https://www.mongodb.com/try/download/community) or MongoDB Atlas (optional if the in-memory fallback is configured)

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd shopsphere
```

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Configure Environment Variables

Create a `.env` file in the project root directory.

Add the following configuration:

```env
PORT=5000
NODE_ENV=development

MONGODB_URI=mongodb://localhost:27017/shopsphere

JWT_SECRET=your_own_long_random_secret
JWT_EXPIRE=30d
COOKIE_EXPIRE=30
```

**Important:** Never expose your actual JWT secret or production environment variables on GitHub.

### Step 4: Seed the Database

Populate the database with sample products and demo accounts:

```bash
npm run seed
```

This creates:

- 25+ sample products.
- 7 product categories.
- Demo administrator account.
- Demo customer account.

### Step 5: Start the Development Server

```bash
npm run dev
```

### Step 6: Open the Application

Open your browser and visit:

```text
http://localhost:5000
```

🎉 Congratulations! ShopSphere is now running locally.

---

## 🔑 Demo Credentials

Use the following accounts to explore the application after successfully running the database seeder.

| Account | Email | Password |
|---|---|---|
| Administrator | admin@shopsphere.com | adminpassword123 |
| Customer | user@shopsphere.com | userpassword123 |

**Note:** These credentials are intended only for local development and demonstration. Do not use them in production.

---

## 📡 API Documentation

ShopSphere provides RESTful APIs for authentication, product management, shopping carts, and order processing.

### 🔐 Authentication APIs

Base URL: `/api/auth`

| Method | Endpoint | Description | Access |
|---|---|---|---|
| POST | `/register` | Register new user | Public |
| POST | `/login` | Authenticate user | Public |
| POST | `/logout` | Logout user | Public |
| GET | `/profile` | Get user profile | Private |
| PUT | `/profile` | Update user profile | Private |
| POST | `/wishlist/:productId` | Toggle wishlist item | Private |

### 📦 Product APIs

Base URL: `/api/products`

| Method | Endpoint | Description | Access |
|---|---|---|---|
| GET | `/` | Get all products | Public |
| GET | `/featured` | Get featured products | Public |
| GET | `/categories` | Get product categories | Public |
| GET | `/:id` | Get product details | Public |
| POST | `/` | Create new product | Admin |
| PUT | `/:id` | Update product | Admin |
| DELETE | `/:id` | Delete product | Admin |

### 🛒 Cart APIs

Base URL: `/api/cart`

| Method | Endpoint | Description | Access |
|---|---|---|---|
| GET | `/` | Get shopping cart | Private |
| POST | `/` | Add item or update quantity | Private |
| PUT | `/item` | Update cart item quantity | Private |
| DELETE | `/item/:productId` | Remove cart item | Private |
| DELETE | `/` | Clear shopping cart | Private |

### 📋 Order APIs

Base URL: `/api/orders`

| Method | Endpoint | Description | Access |
|---|---|---|---|
| POST | `/` | Place new order | Private |
| GET | `/` | Get user order history | Private |
| GET | `/:id` | Get specific order details | Private |

**Access Levels:**

- 🌐 Public: Accessible without authentication.
- 🔒 Private: Requires authenticated user.
- 👑 Admin: Requires administrator privileges.

---

## 🔐 Security Features

ShopSphere implements several security-related mechanisms:

- JWT-based authentication.
- Password hashing using Bcrypt.
- Protected API routes.
- Role-based authorization.
- HttpOnly cookie support.
- Request validation using express-validator.
- Environment variable management.
- Centralized error-handling middleware.
- Controlled access to administrative endpoints.

---

## 🧪 Testing Checklist

The following functionality is included in the project evaluation checklist.

### Authentication
- [x] User registration.
- [x] User login and logout.
- [x] JWT authentication.
- [x] Protected routes.
- [x] Admin authorization.
- [x] Profile management.

### Product Management
- [x] 25+ sample products.
- [x] Seven product categories.
- [x] Product search.
- [x] Category filtering.
- [x] Price filtering.
- [x] Rating filtering.
- [x] Sorting functionality.
- [x] Pagination.

### Shopping Cart
- [x] Add products to cart.
- [x] Update product quantities.
- [x] Remove products.
- [x] Dynamic price calculations.
- [x] Guest cart persistence.
- [x] Guest cart synchronization.

### Orders & Checkout
- [x] Multi-step checkout.
- [x] Shipping address collection.
- [x] Payment method selection.
- [x] Order placement.
- [x] Stock decrementing.
- [x] Order history.
- [x] Order status tracking.

### User Interface
- [x] Responsive design.
- [x] Light and Dark themes.
- [x] Toast notifications.
- [x] Interactive product cards.
- [x] Mobile navigation.
- [x] Theme persistence.

---

## 🚀 Future Enhancements

The following improvements can be considered for future versions:

- [ ] Integration with Razorpay and Stripe payment gateways.
- [ ] Google and GitHub social login.
- [ ] Product reviews and verified customer ratings.
- [ ] Advanced admin analytics dashboard.
- [ ] Real-time order tracking.
- [ ] Email notifications for order updates.
- [ ] Product recommendations using AI.
- [ ] Coupon and discount management.
- [ ] Docker-based deployment.
- [ ] Automated testing using Jest or Supertest.
- [ ] Cloud deployment and CI/CD integration.

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome!

If you would like to contribute:

1. Fork this repository.
2. Create a new feature branch.

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes.

```bash
git commit -m "Add: YourFeature"
```

4. Push your branch.

```bash
git push origin feature/YourFeature
```

5. Open a Pull Request.

---

## 📜 License

This project is distributed under the **MIT License**.

You are free to use, modify, and distribute this project according to the terms of the license.

---

## 👨‍💻 Author

### Prathamesh B. Patil

Engineering Student | AI & ML | Full Stack Development | Data Analytics

Interested in building practical applications using modern web technologies, Artificial Intelligence, and Machine Learning.

<p align="left">
  <a href="https://github.com/Shree0802">
    <img src="https://img.shields.io/badge/GitHub-Profile-181717?style=for-the-badge&logo=github" />
  </a>
  <a href="https://www.linkedin.com/in/prathamesh-patil1718">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin" />
  </a>
</p>

---

<p align="center">
  ⭐ If you find ShopSphere interesting, consider giving this repository a star!
</p>

<p align="center">
  <b>Built with ❤️ using HTML, CSS, JavaScript, Node.js, Express.js & MongoDB.</b>
</p>

<p align="center">
  © 2026 ShopSphere | Developed for CodeAlpha Full Stack Development Internship
</p>
